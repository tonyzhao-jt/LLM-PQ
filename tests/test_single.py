import os
import torch 
from torch.distributed import rpc 
import torch.distributed as dist
from time import perf_counter
from transformers import AutoTokenizer
from transformers import LogitsProcessorList

from qllm.models.OPT import OPTForCausalLMSeq
from qllm.utils import (
    to_device_recursive,
)

from llm_pq import (
    init_random_seed,
    fetch_prompts
)

from llm_pq.rpc import (
    init_env, DistConfig, set_device_map,
    DistRpcContext
)
from llm_pq.logger import init_logger
logger = init_logger(__name__)
from llm_pq.thread import ThreadSafeCounter
from llm_pq.partitioner import get_shard_strategy
from llm_pq.pipe import (
    dist_rpc_pipeline_factory
)

import lptorch

results_counter = ThreadSafeCounter()

final_result = {}
request_input_ids = {}
request_logit_processor = {}
request_loop_counter = {}
def handle_results(final_intermediate_result) -> None:
    request_id = final_intermediate_result[-1]
    request_loop_counter[request_id] += 1
    results_counter.add(1)
    # get original input id
    input_ids = request_input_ids[request_id]
    logits_processor = request_logit_processor[request_id]
    # generate new tokens
    outputs = model_pre_and_post.postprocess(final_intermediate_result, None)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if request_loop_counter[request_id] < num_tokens_to_generate:
        request_input_ids[request_id] = new_input_ids
        request_token = model_pre_and_post.preprocess_one_token(new_input_ids, next_tokens, use_cache=True, request_id=request_id)
        logger.info(f"Request id {request_id} done for token {request_loop_counter[request_id]}")
        master_pipeline.enqueue_tensor(to_device_recursive(request_token, 'cpu'))


master_pipeline = None
def run_pipeline_rpc(model_cpu:list, tokenizer, dist_cfg: DistConfig, chunk:int=1, sharding_strategy=None) -> None:
    global master_pipeline
    """Run the pipeline using RPC communication."""
    rpc_opts = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128, rpc_timeout=60 * 100) # the loading of weight takes a lot of time
    rank = dist_cfg.rank
    data_rank = 0 # by default, use rank 0 as the data rank
    # verify the scheduling is ok to be set
    if rank == 0:
        if not sharding_strategy:
            sharding_strategy = get_shard_strategy(model_cpu)
        else:
            model_cpu._verify_shard_strategy(sharding_strategy)  
    set_device_map(rank, device_mesh, sharding_strategy, rpc_opts)
    # based on the schedule and device_mesh, determines the communication type
    with DistRpcContext((f"worker{rank}",),
                        { 'world_size': dist_cfg.world_size,
                          'rank': rank,
                          'rpc_backend_options': rpc_opts}
                       ) as dist_ctx:
        
        if rank == 0: # master, process some data
            
            # create pipeline
            pipeline = dist_rpc_pipeline_factory(model_cpu, sharding_strategy, device_mesh, infer_configs, rank, handle_results)
            master_pipeline = pipeline
            # prepare test data
            def prepare_input(batched_ids, request_id):
                batched_ids = to_device_recursive(dict(batched_ids), 'cuda:0')
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
            

            data_chunks = []
            for i in range(request_numbers):
                batched_ids = tokenizer.batch_encode_plus(fetch_prompts(bs_token, prompt_length), padding='max_length', max_length=prompt_length, return_tensors="pt")
                request_token = prepare_input(batched_ids, request_id=i)
                data_chunks.append(request_token)
            # print("chunk size", get_iter_variable_size(data_chunks, unit='MB'))
            batch_size = len(data_chunks)
            print("pipeline")

            # fake the data chunks for test
            # TODO: make it real chunks
            # data_chunks = torch.chunk(sample_data_batch, chunk, dim=0)
            # other_decode_params = master_model.other_decode_params
            # dist_ctx.cmd_broadcast(handle_cmd, CMD_SCHED, other_decode_params)
        else:
            print("worker")
            # communicate to get other_decode_params
            # logger.info("Waiting for other params")
            # other_decode_params = sched_q.get()
            # assign the decode params to the corresponding refs
        
        rpc.api._barrier([f"worker{i}" for i in range(dist_cfg.world_size)])
        if rank == data_rank:
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

            for request_id, input_id_finally in request_input_ids.items():
                ouput_token = tokenizer.batch_decode(input_id_finally, skip_special_tokens=True)
                print(f"request {request_id} output token {ouput_token}")


import pickle
from qllm.models.OPT.opt import model_cards
if __name__ == '__main__':

    # load the LLM from QLLM
    # with weight
    # loaded_llm_cpu = OPTForCausalLMSeq.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")


    # test case
    model_size = "350m"
    config = model_cards[model_size]
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    loaded_llm_cpu = OPTForCausalLMSeq._from_config(config, torch_dtype=torch.float16)

    sharding_strategy = {
        0: {},
        1: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
            2: {'shard': [0, 1], 'bits': [16, 16]},
            3: {'shard': [0, 1], 'bits': [16, 16]},
            4: {'shard': [0, 1], 'bits': [16, 16]},
            5: {'shard': [0, 1], 'bits': [16, 8]},
            6: {'shard': [0, 1], 'bits': [16, 16]},
            7: {'shard': [0, 1], 'bits': [16, 16]},
            8: {'shard': [0], 'bits': [16]},
        },
        2: {
            8: {'shard': [1], 'bits': [16]},
            9: {'shard': [0,1], 'bits': [16, 16]},
            10: {'shard': [0,1], 'bits': [8, 16]},
            11: {'shard': [0,1], 'bits': [16, 16]},
            # 350M
            12: {'shard': [0,1], 'bits': [16, 16]},
            13: {'shard': [0,1], 'bits': [16, 16]},
            14: {'shard': [0,1], 'bits': [8, 16]},
            15: {'shard': [0,1], 'bits': [16, 16]},
            16: {'shard': [0,1], 'bits': [16, 16]},
            17: {'shard': [0,1], 'bits': [16, 8]},
        },
        3:{
            18: {'shard': [0,1], 'bits': [16, 16]},
            19: {'shard': [0,1], 'bits': [16, 16]},
            20: {'shard': [0,1], 'bits': [8, 16]},
            21: {'shard': [0,1], 'bits': [16, 16]},
            22: {'shard': [0,1], 'bits': [16, 16]}, 
            23: {'shard': [0,1], 'bits': [16, 16]},
        }
    }
    # control the token generation
    num_tokens_to_generate = 50
    prompt_length = 512
    bs_token = 4 # how many sentence in a batch
    request_numbers = 4 # how many requests
    chunk = 1

    infer_configs = (bs_token, prompt_length, num_tokens_to_generate, request_numbers)
    loaded_llm_cpu._verify_shard_strategy(sharding_strategy)

    # init env
    seed = 42
    init_random_seed(seed)
    dist_cfg, device_mesh = init_env()
    assert dist_cfg.world_size > 1, "world size should be larger than 1, else single device"

    if dist_cfg.rank == 0:
        model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
        model_pre_and_post = model_pre_and_post.cuda()

    run_pipeline_rpc(loaded_llm_cpu, tokenizer, dist_cfg, chunk=chunk, sharding_strategy=sharding_strategy)
