import os
import pickle

import torch 
import torch.distributed as dist
from time import perf_counter
from transformers import AutoTokenizer
from transformers import LogitsProcessorList

import lptorch 

from qllm.utils import (
    to_device_recursive
)
from qllm.models import create_empty_model
from qllm.scheduler import DSScheduler

CMD_STOP = 0
CMD_SCHED = 1

import qpipe
from qpipe import (
    init_random_seed,
    fetch_prompts
)
from qpipe.logger import logger
# dist
from qpipe.p2p import (
    init_env, DistP2pContext,
    handle_cmd, stop_event,
    create_device_mesh,
    new_nccl_group
)
from qpipe.thread import ThreadSafeCounter
from qpipe.p2p.dist_pipe import (
    dist_p2p_pipeline_stage_factory, SimpleQueueThread, ConditionQueue
)

import queue
import time

master_stage_context = None
results_counter = ThreadSafeCounter()
lock_queue = None
work_queue = None
simple_queue_thread = None
final_result = {}
request_input_ids = {}
request_logit_processor = {}
request_loop_counter = {}

def handle_results(final_intermediate_result) -> None:
    request_id = final_intermediate_result[-2]
    if isinstance(request_id, torch.Tensor):
        request_id = request_id.item()
    results_counter.add(1)
    # print(results_counter._value)
    # get original input id
    input_ids = request_input_ids[request_id]
    logits_processor = request_logit_processor[request_id]
    # generate new tokens
    final_intermediate_result = to_device_recursive(final_intermediate_result, 'cuda:0')
    outputs = model_pre_and_post.postprocess(final_intermediate_result, None)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    flag, concat_tokens = ds_scheduler.pass_scheduler(request_id, next_tokens)
    if flag:
        # print(flag, concat_tokens.shape)
        request_loop_counter[request_id] += 1
        new_input_ids = torch.cat([input_ids, concat_tokens], dim=-1)

        # new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if request_loop_counter[request_id] < num_tokens_to_generate:
            if request_loop_counter[request_id] == 0:
                return # do nothing
            request_input_ids[request_id] = new_input_ids
            request_token = model_pre_and_post.preprocess_one_token(new_input_ids, concat_tokens, use_cache=True, request_id=request_id)
            logger.info(f"Request id {request_id} done for token {request_loop_counter[request_id]}")
            if args.nccl:
                payload = request_token
            else:
                payload = to_device_recursive(request_token, 'cpu')
            with work_queue.condition:
                while work_queue.full():
                    work_queue.condition.wait()
                work_queue.put(payload)
                work_queue.condition.notify_all()


def set_input_ids_globals(request_id, p_input_ids):
    generation_config = model_pre_and_post.generation_config
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


from datetime import timedelta


def init_tokenizer():
    if model_name == 'opt':
        return AutoTokenizer.from_pretrained("facebook/opt-66b")
    elif model_name == 'bloom':
        return AutoTokenizer.from_pretrained("bigscience/bloom")
    

def run_inf(stage_ctx, input_id_dict, data_chunks, sample_num=None):
    global lock_queue, work_queue, simple_queue_thread, num_tokens_to_generate
    if sample_num is not None:
        num_tokens_to_generate = sample_num
    lock_queue = ConditionQueue(maxsize=1)
    work_queue = ConditionQueue(maxsize=chunk_size)
    simple_queue_thread = SimpleQueueThread(lock_queue, work_queue, master_stage_context.enqueue_tensor)
    simple_queue_thread.start() # start
    ds_scheduler.reset_status()
    # set global vars.
    for chunk_idx, input_id in enumerate(input_id_dict.values()):
        set_input_ids_globals(chunk_idx, input_id)
    prefill_cnt = len(data_chunks)
    results_counter.set(0) # reset

    tik_data = perf_counter()
    logger.info("start pipe data")
    start_count = results_counter.value
    # this only launch the tasks but not actually finish the tasks.
    for data_chunk in data_chunks:
        stage_ctx.enqueue_tensor(data_chunk)
    
    # unlock the queue
    with lock_queue.condition:
        lock_queue.put(1)
        lock_queue.condition.notify_all()
    results_counter.wait_gte(start_count + prefill_cnt + chunk_size * (num_tokens_to_generate - 1))

    tok_data = perf_counter()
    latency = tok_data - tik_data
    # throughput  = bs * N(token generated) / latency
    throughput = bs_token / latency
    token_throughput = throughput * num_tokens_to_generate
    logger.info("Latency is %f, throughput is %f", latency, throughput)
    logger.info('Token throughput is %f', token_throughput)


def run_pipeline_p2p(loaded_llm_cpu, dist_cfg, sharding_strategy=None):
    global master_stage_context
    global simple_queue_thread, lock_queue, work_queue
    rank = dist_cfg.rank
    local_rank = dist_cfg.local_rank
    data_rank = 0 # by default, use rank 0 as the data rank
    world_size = dist_cfg.world_size
    # verify the scheduling is ok to be set
    if rank == 0:
        if not sharding_strategy:
            raise ValueError("sharding strategy is not set")
        else:
            loaded_llm_cpu._verify_shard_strategy(sharding_strategy)  
    # communication group init with gloo
    with DistP2pContext(('gloo',), { 'world_size': world_size, 'rank': rank, 'timeout': timedelta(seconds=1800)}, handle_cmd) \
        as dist_ctx:

        input_id_dict, data_chunks = None, None
        if rank == data_rank:
            device = torch.device(f'cuda:{local_rank}')
            # init tokenizer
            tokenizer = init_tokenizer()
            data_chunks = []
            sample_texts = fetch_prompts(bs_token, prompt_length)
            sample_text_dict = ds_scheduler.split_list_of_prompts(sample_texts)
            prefill_bs_indexes = ds_scheduler.create_ds_indexes()
            for request_id, sample_text_list in sample_text_dict.items():
                sample_text_dict[request_id] = to_device_recursive([dict(tokenizer.batch_encode_plus(text, padding='max_length', max_length=prompt_length, return_tensors="pt")) for text \
                                in sample_text_list], device)
            data_chunks = []
            input_id_dict = {}
            for request_id, prefill_bs_indexes in prefill_bs_indexes.items():
                for idx, prefill_bs_index in enumerate(prefill_bs_indexes):
                    current_sub_request_batch_ids = sample_text_dict[request_id][idx]
                    if request_id not in input_id_dict:
                        input_id_dict[request_id] = current_sub_request_batch_ids['input_ids']
                    else:
                        input_id_dict[request_id] = torch.cat([input_id_dict[request_id], current_sub_request_batch_ids['input_ids']], dim=0)
                    # print(current_sub_request_batch_ids['input_ids'].shape)
                    request_token = model_pre_and_post.preprocess(**current_sub_request_batch_ids, use_cache=True, request_id=request_id, batch_index=prefill_bs_index)
                    if not args.nccl:
                        request_token = to_device_recursive(request_token, 'cpu')
                    data_chunks.append(request_token)
        
        # get stage
        if rank not in sharding_strategy:
            stage_id = None
        else:
            stage_ranks = sorted(list(sharding_strategy.keys()))
            stage_id = stage_ranks.index(rank)
            # shard model
            print("rank {} is in stage {}".format(rank, stage_id))
        

        # sharded module init
        shard_config = sharding_strategy[rank]
        module = loaded_llm_cpu
        module._shard_model_current(shard_config, f'cuda:{local_rank}')
        print(f"Stage {stage_id} module sharded")
        for chunk_id in range(chunk_size):
            # init kv cache for each decoder bs
            # the prefill stage will use the same cache created by decoder
            module.init_kv_cache(decoder_bss[chunk_id], prompt_length, max_tokens_to_generate, chunk_id)
        print(f"Stage {stage_id} kv initialized")
        module.eval()
        module.on_device = f'cuda:{local_rank}'
        dist.barrier() # wait all device sharded finished.

        # new a nccl group
        if args.nccl:
            nccl_group = new_nccl_group()
            qpipe._globals.__PIPELINE__MODEL__PARALLEL__GROUP__ = nccl_group
            qpipe._globals.__DEVICE__INDEX__ = local_rank
        with dist_p2p_pipeline_stage_factory(stage_ranks, data_rank, rank, stage_id, module,
                                                        handle_results) as stage_ctx:
            
            
            if rank == data_rank:
                master_stage_context = stage_ctx
                original_num_tokens_to_generate = num_tokens_to_generate
                run_inf(stage_ctx, input_id_dict, data_chunks, sample_num=warmup_tokens)
                dist.barrier()
                module._reset_kv_status()
                
                dist.barrier()
                run_inf(stage_ctx, input_id_dict, data_chunks, sample_num=original_num_tokens_to_generate)
                dist_ctx.cmd_broadcast(CMD_STOP)
                # join the queue thread
                simple_queue_thread.stop()
                simple_queue_thread.join()
                stop_event.set()
            else:
                dist.barrier()
                module._reset_kv_status()
                dist.barrier()
                stop_event.wait()
    pass

import argparse
def parse_args():
    # add argparser for model name and model_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="350m", help="model size")
    parser.add_argument("--model_name", type=str, default="opt", help="model name")
    parser.add_argument("--bs_token", type=int, default=32, help="Global batch size for token")
    parser.add_argument("--prompt_length", type=int, default=512, help="prompt length")
    parser.add_argument("--max_tokens_to_generate", type=int, default=100, help="number of tokens to generate")
    parser.add_argument("--num_tokens_to_generate", type=int, default=100, help="number of tokens to generate")
    parser.add_argument("--warmup_tokens", type=int, default=2, help="warmup")
    parser.add_argument("--nccl", action='store_true', default=False, help="use nccl")
    parser.parse_args()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set env
    os.environ['SET_DECODERS_META'] = "1"
    os.environ['PERF_MODE'] = "1"
    args = parse_args()
    # test case
    model_name = args.model_name
    model_size = args.model_size
    loaded_llm_cpu = create_empty_model(model_name, model_size)

    # load the fake calibration data
    caliber = lptorch.inner_caliber
    caliber.set_fake() 
    caliber.load_fake_calib_data(f'fake_calib_{model_name}_{model_size}.pkl')
  
    sharding_strategy = {
        0: {},
        1: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
            2: {'shard': [0, 1], 'bits': [8, 16]},
            3: {'shard': [0, 1], 'bits': [16, 16]},
            4: {'shard': [0, 1], 'bits': ['8:tc', '8:tc']},
            5: {'shard': [0, 1], 'bits': ['8:tc-li', 8]},
            6: {'shard': [0, 1], 'bits': ['8:tc', '8:tc']},
            7: {'shard': [0, 1], 'bits': ['8:tc-li', 16]},
            8: {'shard': [0], 'bits': [16]},
        },
        2: {
            8: {'shard': [1], 'bits': [16]},
            9: {'shard': [0,1], 'bits': ['8:tc-li', 8]},
            10: {'shard': [0,1], 'bits': [8, 16]},
            11: {'shard': [0,1], 'bits': [2, 16]},
            # 350M
            12: {'shard': [0,1], 'bits': ['8:tc-li', 16]},
            13: {'shard': [0,1], 'bits': ['8:tc-li', 4]},
            14: {'shard': [0,1], 'bits': [8, 16]},
            15: {'shard': [0,1], 'bits': ['8:tc-li', 16]},
            16: {'shard': [0,1], 'bits': ['8:tc-li', 8]},
            17: {'shard': [0,1], 'bits': [16, 8]},
        },
        3:{
            18: {'shard': [0,1], 'bits': [16, '8:tc-li']},
            19: {'shard': [0,1], 'bits': ['8:tc-li', 16]},
            20: {'shard': [0,1], 'bits': [8, '8:tc-li']},
            21: {'shard': [0,1], 'bits': [4, '8:tc-li']},
            22: {'shard': [0,1], 'bits': ['8:tc-li', 16]}, 
            23: {'shard': [0,1], 'bits': [16,'8:tc-li']},
        }
    }

    # control the token generation
    max_tokens_to_generate = args.max_tokens_to_generate
    num_tokens_to_generate = args.num_tokens_to_generate
    warmup_tokens = args.warmup_tokens
    prompt_length = args.prompt_length
    bs_token = args.bs_token # how many sentence in a batch

    prefill_bs = 1
    decoder_bss = [8,8, 8, 8]
    chunk_size = len(decoder_bss)
    ds_scheduler = DSScheduler(prefill_bs, decoder_bss)

    infer_configs = (bs_token, prompt_length, num_tokens_to_generate, chunk_size)
    loaded_llm_cpu._verify_shard_strategy(sharding_strategy)

    # init env
    seed = 42
    init_random_seed(seed)
    dist_cfg, hard_device_mesh = init_env()
    assert dist_cfg.world_size > 1, "world size should be larger than 1, else single device"

    if dist_cfg.rank == 0:
        model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
        model_pre_and_post = model_pre_and_post.cuda()

    run_pipeline_p2p(loaded_llm_cpu, dist_cfg, sharding_strategy=sharding_strategy)
    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(
    #     activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
    #     record_shapes=True,
    #     with_stack=True) as prof:
    #     with record_function("model_inference"):
    #         run_pipeline_p2p(loaded_llm_cpu, dist_cfg, sharding_strategy=sharding_strategy)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # # prof.export_chrome_trace("trace.json")


    