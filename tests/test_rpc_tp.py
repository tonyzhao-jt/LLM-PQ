import os
import pickle
import torch 
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import rpc 

from time import perf_counter
import copy 


from transformers import AutoTokenizer
from transformers import LogitsProcessorList, StoppingCriteriaList

CMD_STOP = 0
CMD_SCHED = 1

import lptorch 

from qllm.utils import (
    to_device_recursive
)

from qllm.models import create_empty_model
import qllm.tp.utils as qllm_tp_utils

import qpipe 
from qpipe import (
    init_random_seed,
    fetch_prompts
)
from qpipe.logger import logger

from qpipe.rpc import (
    init_env, 
    DistConfig, set_device_map,
    DistRpcContext,
    stop_event,
    ConditionQueue
)

from qpipe.logger import logger
from qpipe.thread import ThreadSafeCounter
from qpipe.partitioner import get_shard_strategy
from qpipe.pipe import (
    dist_rpc_pipeline_factory
)

import threading

results_counter = ThreadSafeCounter()
final_result = {}
request_input_ids = {}
request_logit_processor = {}
request_loop_counter = {}
model_pre_and_post = None

# cq = ConditionQueue(maxsize=1)

def handle_results(final_intermediate_result) -> None:
    # cq.put(final_intermediate_result)
    # with cq.condition:
    #     while cq.empty():
    #         cq.condition.wait()
    #     final_intermediate_result = cq.get()
    #     cq.condition.notify_all()
    request_id = final_intermediate_result[-1]
    request_loop_counter[request_id] += 1
    results_counter.add(1)
    # get original input id
    input_ids = request_input_ids[request_id]
    logits_processor = request_logit_processor[request_id]
    # generate new tokens
    final_intermediate_result = to_device_recursive(final_intermediate_result, f'cuda:{qpipe._globals.__DEVICE__INDEX__}')
    outputs = model_pre_and_post.postprocess(final_intermediate_result, None)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if request_loop_counter[request_id] < num_tokens_to_generate:
        request_input_ids[request_id] = new_input_ids
        request_token = model_pre_and_post.preprocess_one_token(new_input_ids, next_tokens, use_cache=True, request_id=request_id)
        logger.info(f"Request id {request_id} done for token {request_loop_counter[request_id]}")
        # enqueue the request_token tensor
        master_stage_context.enqueue_tensor(to_device_recursive(request_token, 'cpu'))


def empty_callback():
    pass

def create_handle_results_fn(model_pre_and_post):
    def handle_results_in(final_intermediate_result):
        return handle_results(final_intermediate_result, model_pre_and_post)
    return handle_results_in


# prepare test data
def prepare_input(model_pre_and_post, batched_ids, request_id, device):
    batched_ids = to_device_recursive(dict(batched_ids), device)
    generation_config = model_pre_and_post.generation_config
    request_token = model_pre_and_post.preprocess(**batched_ids, use_cache=True, request_id=request_id)
    p_input_ids = batched_ids['input_ids']
    inputs_tensor, model_input_name, model_kwargs = model_pre_and_post._prepare_model_inputs(
        p_input_ids, generation_config.bos_token_id, {}
    )
    input_ids_seq_length = p_input_ids.shape[-1]
    logits_processor = LogitsProcessorList()
    # 8. prepare distribution pre_processing samplers
    logits_processor = model_pre_and_post._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
    )
    final_result[request_id] = None
    request_input_ids[request_id] = p_input_ids
    request_logit_processor[request_id] = logits_processor
    request_loop_counter[request_id] = 0

    return to_device_recursive(request_token, 'cpu')

def init_tokenizer():
    if model_name == 'opt':
        return AutoTokenizer.from_pretrained("facebook/opt-66b")
    elif model_name == 'bloom':
        return AutoTokenizer.from_pretrained("bigscience/bloom")

def get_model_rref():
    return rpc.RRef(qpipe._globals.__CURRENT__SHARDED__MODEL__)


def process_mesh_and_set_env(rank, device_mesh):
    # read device_mesh
    head_stage_ranks = [value[0] for value in device_mesh.values()] # first rank
    all_ranks_involved = list(set(value for sublist in device_mesh.values() for value in sublist))
    for head_stage_id, stage_ranks in device_mesh.items():
        if rank in stage_ranks:
            stage_id = qpipe._globals.__STAGE__ID__ = head_stage_id
        if len(stage_ranks) > 0:
            # init inder qllm tp group
            res = qllm_tp_utils.register_tp_group_and_update_strategy(stage_ranks, sharding_strategy)
            if res is not None:
                _, tp_index, tp_small_world_size = res
                qllm_tp_utils.disable_broadcast() # disable broadcast inside layer, do single broadcast outside (in begining of pipiline stage)
                qpipe._globals.__TP__LOCAL__RANK__ = tp_index
                qpipe._globals.__TENSOR__MODEL__PARALLEL__GROUP__ = qllm_tp_utils.get_tp_group()
                qpipe._globals.__TP__LOCAL__WORLD__SIZE__ = tp_small_world_size
                qpipe._globals.__TP__GROUP__RANKS__ = stage_ranks
                if tp_index == 0:
                    print('init tp group', stage_ranks)
    return stage_id, head_stage_ranks, all_ranks_involved

cur_tp_config = None
def empty_model_tp_configs():
    global cur_tp_config
    cur_tp_config = qllm_tp_utils.get_tp_configs()
    qllm_tp_utils.empty_tp_configs()

def load_model_tp_configs():
    qllm_tp_utils.load_tp_configs(cur_tp_config)
    # logger.info(f"load tp configs {cur_tp_config}")

master_stage_context = None
def run_pipeline_rpc(loaded_llm_cpu, dist_cfg, sharding_strategy=None) -> None:
    global model_pre_and_post
    global master_stage_context
    rank = dist_cfg.rank
    local_rank = dist_cfg.local_rank
    # verify the scheduling is ok to be set
    if rank == 0:
        if not sharding_strategy:
            raise ValueError("sharding strategy is not set")
        else:
            loaded_llm_cpu._verify_shard_strategy(sharding_strategy)

    # init stage
    stage_id, head_stage_ranks, all_ranks_involved = process_mesh_and_set_env(rank, device_mesh)
    qpipe._globals.__DEVICE__INDEX__ = local_rank
    device = torch.device(f'cuda:{local_rank}')
    qpipe._globals.__GLOBAL__RANK__ = rank    
    rpc_opts = rpc.TensorPipeRpcBackendOptions(rpc_timeout=60) # the loading of weight takes a lot of time
    stages_2d, rpc_opts = set_device_map(rank, device_mesh, hard_device_mesh, rpc_opts)

    if stage_id == 0: # first stage is used for data rank.
        model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
        model_pre_and_post = model_pre_and_post.to(device)
        model_pre_and_post.eval()
        dist.barrier(qllm_tp_utils.get_tp_group())
    # module initialization
    dist.barrier() 
    current_stage_ranks = device_mesh[stage_id]
    logger.info("current rank {} on stage {}, TP GROUP {}".format(rank, stage_id, current_stage_ranks))
    if rank in current_stage_ranks:
        shard_config = sharding_strategy[rank]
        module = loaded_llm_cpu
        module._shard_model_current(shard_config, f'cuda:{local_rank}')
        if len(current_stage_ranks) > 0: # TP is initialized
            dist.barrier(qllm_tp_utils.get_tp_group())
        logger.info(f"Stage {stage_id} - {qpipe._globals.__TP__LOCAL__RANK__} module sharded")

        for request_id in range(request_numbers):
            module.init_kv_cache(bs_token, prompt_length, num_tokens_to_generate, request_id)
        logger.info(f"Stage {stage_id} - {qpipe._globals.__TP__LOCAL__RANK__} kv initialized")
        module.eval()
        module.on_device = f'cuda:{local_rank}' # set device for the module
    qpipe._globals.__CURRENT__SHARDED__MODEL__ = module # set module
    dist.barrier()

    # Embedding Execution. Prepare data.
    if stage_id == 0: # first stage is used for data rank.
        # init tokenizer
        tokenizer = init_tokenizer()
        data_chunks = []
        for i in range(request_numbers):
            batched_ids = tokenizer.batch_encode_plus(fetch_prompts(bs_token, prompt_length), padding='max_length', max_length=prompt_length, return_tensors="pt")
            request_token = prepare_input(model_pre_and_post, batched_ids, request_id=i, device=device)
            # request_token = to_device_recursive(request_token, device=device)
            request_token = to_device_recursive(request_token, 'cpu') 
            data_chunks.append(request_token)
        # print("chunk size", get_iter_variable_size(data_chunks, unit='MB'))
        batch_size = len(data_chunks)
        logger.info(f"Stage {stage_id} - {qpipe._globals.__TP__LOCAL__RANK__} data prepared")
    dist.barrier()

    # ALERT: Temporarily disable TP
    empty_model_tp_configs()
    logger.warning("RANK {}: TP is disabled TMP".format(rank))
    dist.barrier()
    # # based on the schedule and device_mesh, determines the communication type
    with DistRpcContext((f"worker{rank}",),
                        { 'world_size': dist_cfg.world_size,
                          'rank': rank,
                          'rpc_backend_options': rpc_opts}
                       ) as dist_ctx:
        if stage_id == 0:
            # stage 2d [i, j]: j: stage, i: tp_index
            i = stages_2d[:, 0].tolist().index(rank)
            assert i != -1, "rank {} not found in stage 0 in 2d stage mesh".format(rank)
            same_row_ranks = stages_2d[i, :].tolist()
            rref_dict = {}
            for row_device in same_row_ranks:
                rref_dict[row_device] = rpc.remote(f"worker{row_device}", get_model_rref)

            pipeline = dist_rpc_pipeline_factory(same_row_ranks, rref_dict, rank, handle_results)
            logger.info("Pipeline Loaded for stage {}-{}".format(stage_id, i))
            # reload tp configs back
            for row_device in same_row_ranks:
                rpc.remote(f"worker{row_device}", load_model_tp_configs)
            master_stage_context = pipeline
            # pipeline.rpc_register_forward_hook(forward_hook_to_cpu)
            # pipeline.rpc_register_forward_pre_hook(forward_pre_hook_to_device)
            tik_data = perf_counter()
            # start results monitoring - see comments in handle_results
            # this call is asynchronous - wait for results to get end-to-end timings
            logger.info("start pipe data")

            start_count = results_counter.value
            # this only launch the tasks but not actually finish the tasks.
            for data_chunk in data_chunks:
                pipeline.enqueue_tensor(data_chunk)
            results_counter.wait_gte(start_count + len(data_chunks) * num_tokens_to_generate)
            tok_data = perf_counter()
            latency = tok_data - tik_data
            # throughput  = bs * N(token generated) / latency
            throughput = batch_size / latency
            token_throughput = throughput * num_tokens_to_generate
            logger.info("Latency is %f, throughput is %f", latency, throughput)
            logger.info('Token throughput is %f', token_throughput)
            dist_ctx.cmd_broadcast(CMD_STOP)
            stop_event.set()
        else:
            stop_event.wait()
#     if rank == 0: # master, process some data
    #         # create pipeline
    #         pipeline = dist_rpc_pipeline_factory(model_cpu, sharding_strategy, device_mesh, infer_configs, rank, handle_results)
    #         master_stage_context = pipeline
    #         # prepare test data
    #         def prepare_input(batched_ids, request_id):
    #             batched_ids = to_device_recursive(dict(batched_ids), 'cuda:0')
    #             generation_config = model_pre_and_post.generation_config
    #             request_token = model_pre_and_post.preprocess(**batched_ids, use_cache=True, request_id=request_id)
    #             p_input_ids = batched_ids['input_ids']
    #             inputs_tensor, model_input_name, model_kwargs = model_pre_and_post._prepare_model_inputs(
    #                 p_input_ids, generation_config.bos_token_id, {}
    #             )
    #             input_ids_seq_length = p_input_ids.shape[-1]
    #             logits_processor = LogitsProcessorList()
    #             # 8. prepare distribution pre_processing samplers
    #             logits_processor = model_pre_and_post._get_logits_processor(
    #                 generation_config=generation_config,
    #                 input_ids_seq_length=input_ids_seq_length,
    #                 encoder_input_ids=inputs_tensor,
    #                 prefix_allowed_tokens_fn=None,
    #                 logits_processor=logits_processor,
    #             )

    #             final_result[request_id] = None
    #             request_input_ids[request_id] = p_input_ids
    #             request_logit_processor[request_id] = logits_processor
    #             request_loop_counter[request_id] = 0

    #             return to_device_recursive(request_token, 'cpu')
            
    #         data_chunks = []
    #         for i in range(request_numbers):
    #             batched_ids = tokenizer.batch_encode_plus(fetch_prompts(bs_token, prompt_length), padding='max_length', max_length=prompt_length, return_tensors="pt")
    #             request_token = prepare_input(batched_ids, request_id=i)
    #             data_chunks.append(request_token)
    #         batch_size = len(data_chunks)
    #         print("pipeline")
    #         # fake the data chunks for test
    #         # TODO: make it real chunks
    #         # data_chunks = torch.chunk(sample_data_batch, chunk, dim=0)
    #         # other_decode_params = master_model.other_decode_params
    #         # dist_ctx.cmd_broadcast(handle_cmd, CMD_SCHED, other_decode_params)
    #     else:
    #         print("worker")
    #         # communicate to get other_decode_params
    #         # logger.info("Waiting for other params")
    #         # other_decode_params = sched_q.get()
    #         # assign the decode params to the corresponding refs
    #     rpc.api._barrier([f"worker{i}" for i in range(dist_cfg.world_size)])
    #     if rank == data_rank:
    #         # pipeline.rpc_register_forward_hook(forward_hook_to_cpu)
    #         # pipeline.rpc_register_forward_pre_hook(forward_pre_hook_to_device)
    #         tik_data = perf_counter()
    #         # start results monitoring - see comments in handle_results
    #         # this call is asynchronous - wait for results to get end-to-end timings
    #         logger.info("start pipe data")
    #         start_count = results_counter.value
    #         # this only launch the tasks but not actually finish the tasks.
    #         for data_chunk in data_chunks:
    #             pipeline.enqueue_tensor(data_chunk)
    #         results_counter.wait_gte(start_count + len(data_chunks) * num_tokens_to_generate)
    #         tok_data = perf_counter()
    #         latency = tok_data - tik_data
    #         throughput = batch_size / latency
    #         token_throughput = throughput * num_tokens_to_generate
    #         logger.info("Latency is %f, throughput is %f", latency, throughput)
    #         logger.info('Token throughput is %f', token_throughput)

    #         for request_id, input_id_finally in request_input_ids.items():
    #             ouput_token = tokenizer.batch_decode(input_id_finally, skip_special_tokens=True)
    #             print(f"request {request_id} output token {ouput_token}")
    # print("pipeline")
    # print("rank", rank, \
    #       "torch.cuda.memory_cached: %fGB"%(torch.cuda.memory_cached(0)/1024/1024/1024), \
    #         "torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024), \
    #         "torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024), \
    #         "torch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated(0)/1024/1024/1024))

import argparse
def parse_args():
    # add argparser for model name and model_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="350m", help="model size")
    parser.add_argument("--model_name", type=str, default="opt", help="model name")
    parser.add_argument("--bs_token", type=int, default=4, help="batch size for token")
    parser.add_argument("--prompt_length", type=int, default=512, help="prompt length")
    parser.add_argument("--num_tokens_to_generate", type=int, default=100, help="number of tokens to generate")
    parser.add_argument("--request_numbers", type=int, default=4, help="number of requests")
    parser.parse_args()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set env
    os.environ['SET_DECODERS_META'] = "1"
    args = parse_args()
    # test case
    model_name = args.model_name
    model_size = args.model_size
    loaded_llm_cpu = create_empty_model(model_name, model_size)

    # load the fake calibration data
    caliber = lptorch.inner_caliber
    caliber.set_fake() 
    caliber.load_fake_calib_data(f'fake_calib_{model_name}_{model_size}.pkl')

    # 2d device_mesh
    # stage_id: [tp_1, tp_2]
    device_mesh = {
        0: [0, 1],
        1: [2, 3]
    }

    sharding_strategy = {
        0: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
            2: {'shard': [0, 1], 'bits': [8, 16]},
            3: {'shard': [0, 1], 'bits': [16, 16]},
            4: {'shard': [0, 1], 'bits': [16, '8:tc-li']},
            5: {'shard': [0, 1], 'bits': [16, 8]},
            6: {'shard': [0, 1], 'bits': [16, 16]},
            7: {'shard': [0, 1], 'bits': [16, 16]},
            8: {'shard': [0, 1], 'bits': [16, 16]},
        },
        1: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
            2: {'shard': [0, 1], 'bits': [8, 16]},
            3: {'shard': [0, 1], 'bits': [16, 16]},
            4: {'shard': [0, 1], 'bits': [16, '8:tc-li']},
            5: {'shard': [0, 1], 'bits': [16, 8]},
            6: {'shard': [0, 1], 'bits': [16, 16]},
            7: {'shard': [0, 1], 'bits': [16, 16]},
            8: {'shard': [0, 1], 'bits': [16, 16]},
        },
        2: {
            9: {'shard': [0,1], 'bits': [16, 8]},
            10: {'shard': [0,1], 'bits': [8, 16]},
            11: {'shard': [0,1], 'bits': [2, 16]},
            # 350M
            12: {'shard': [0,1], 'bits': [16, 16]},
            13: {'shard': [0,1], 'bits': [16, 4]},
            14: {'shard': [0,1], 'bits': [8, 16]},
            15: {'shard': [0,1], 'bits': [16, 16]},
            16: {'shard': [0,1], 'bits': [16, 8]},
            17: {'shard': [0,1], 'bits': [16, 8]},
            18: {'shard': [0,1], 'bits': [16, 16]},
            19: {'shard': [0,1], 'bits': [16, 16]},
            20: {'shard': [0,1], 'bits': [8, 16]},
            21: {'shard': [0,1], 'bits': [4, 16]},
            22: {'shard': [0,1], 'bits': [16, 16]}, 
            23: {'shard': [0,1], 'bits': [16, 16]},
        },
        3:{
            9: {'shard': [0,1], 'bits': [16, 8]},
            10: {'shard': [0,1], 'bits': [8, 16]},
            11: {'shard': [0,1], 'bits': [2, 16]},
            # 350M
            12: {'shard': [0,1], 'bits': [16, 16]},
            13: {'shard': [0,1], 'bits': [16, 4]},
            14: {'shard': [0,1], 'bits': [8, 16]},
            15: {'shard': [0,1], 'bits': [16, 16]},
            16: {'shard': [0,1], 'bits': [16, 8]},
            17: {'shard': [0,1], 'bits': [16, 8]},
            18: {'shard': [0,1], 'bits': [16, 16]},
            19: {'shard': [0,1], 'bits': [16, 16]},
            20: {'shard': [0,1], 'bits': [8, 16]},
            21: {'shard': [0,1], 'bits': [4, 16]},
            22: {'shard': [0,1], 'bits': [16, 16]}, 
            23: {'shard': [0,1], 'bits': [16, 16]},
        }
    }

    # control the token generation
    num_tokens_to_generate = args.num_tokens_to_generate
    prompt_length = args.prompt_length
    bs_token = args.bs_token # how many sentence in a batch
    request_numbers = args.request_numbers # how many requests

    infer_configs = (bs_token, prompt_length, num_tokens_to_generate, request_numbers)
    loaded_llm_cpu._verify_shard_strategy(sharding_strategy)

    # init env
    seed = 42
    init_random_seed(seed)
    dist_cfg, hard_device_mesh = init_env()
    assert dist_cfg.world_size > 1, "world size should be larger than 1, else single device"

    
    run_pipeline_rpc(loaded_llm_cpu, dist_cfg, sharding_strategy=sharding_strategy)
