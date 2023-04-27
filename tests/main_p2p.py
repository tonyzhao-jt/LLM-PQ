import os
import pickle

import torch 
import torch.distributed as dist
from time import perf_counter
from transformers import AutoTokenizer
from transformers import LogitsProcessorList

import lptorch 

from qllm.models.OPT import OPTForCausalLMSeq
from qllm.utils import (
    to_device_recursive
)

from qllm.models.OPT import OPTForCausalLMSeq
from qllm.models import opt

from qllm.models.BLOOM import BloomForCausalLMSeq
from qllm.models import bloom

CMD_STOP = 0
CMD_SCHED = 1

from qpipe import (
    init_random_seed,
    fetch_prompts
)
from qpipe.logger import logger
# dist
from qpipe.p2p import (
    init_env, DistP2pContext,
    handle_cmd, stop_event,
    create_device_mesh
)
from qpipe.thread import ThreadSafeCounter
from qpipe.p2p.dist_pipe import (
    dist_p2p_pipeline_stage_factory
)

master_stage_context = None

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
    final_intermediate_result = to_device_recursive(final_intermediate_result, 'cuda:0')
    outputs = model_pre_and_post.postprocess(final_intermediate_result, None)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if request_loop_counter[request_id] < num_tokens_to_generate:
        request_input_ids[request_id] = new_input_ids
        request_token = model_pre_and_post.preprocess_one_token(new_input_ids, next_tokens, use_cache=True, request_id=request_id)
        logger.info(f"Request id {request_id} done for token {request_loop_counter[request_id]}")
        master_stage_context.enqueue_tensor(to_device_recursive(request_token, 'cpu'))


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

from datetime import timedelta

def init_tokenizer():
    if model_name == 'opt':
        return AutoTokenizer.from_pretrained("facebook/opt-66b")
    elif model_name == 'bloom':
        return AutoTokenizer.from_pretrained("bigscience/bloom")

def run_pipeline_p2p(loaded_llm_cpu, dist_cfg, sharding_strategy=None):
    global master_stage_context
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
    with DistP2pContext(('gloo',), { 'world_size': world_size, 'rank': rank, 'timeout': timedelta(seconds=60 * 1000)}, handle_cmd) \
        as dist_ctx:
        device_mesh = create_device_mesh(rank, local_rank, world_size)
        # print("dist context created for: ", rank)
        # print(device_mesh) 

        if rank == data_rank:
            tokenizer = init_tokenizer()
            data_chunks = []
            for i in range(request_numbers):
                batched_ids = tokenizer.batch_encode_plus(fetch_prompts(bs_token, prompt_length), padding='max_length', max_length=prompt_length, return_tensors="pt")
                request_token = prepare_input(batched_ids, request_id=i)
                data_chunks.append(request_token)
            # print("chunk size", get_iter_variable_size(data_chunks, unit='MB'))
            batch_size = len(data_chunks)
            print("Pipeline Data Loaded")
            
        
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
        for request_id in range(request_numbers):
            module.init_kv_cache(bs_token, prompt_length, num_tokens_to_generate, request_id)
        print(f"Stage {stage_id} kv initialized")
        module.eval()
        module.on_device = f'cuda:{local_rank}'
        dist.barrier() # wait all device sharded finished.

        with dist_p2p_pipeline_stage_factory(stage_ranks, data_rank, rank, stage_id, module,
                                                        handle_results) as stage_ctx:

            if rank == data_rank:
                master_stage_context = stage_ctx
                # pipeline.rpc_register_forward_hook(forward_hook_to_cpu)
                # pipeline.rpc_register_forward_pre_hook(forward_pre_hook_to_device)
                tik_data = perf_counter()
                # start results monitoring - see comments in handle_results
                # this call is asynchronous - wait for results to get end-to-end timings
                logger.info("start pipe data")
                start_count = results_counter.value
                # this only launch the tasks but not actually finish the tasks.
                for data_chunk in data_chunks:
                    stage_ctx.enqueue_tensor(data_chunk)
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
        
    pass


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
    parser.add_argument("--method", type=str, default="pipeedge", help="method of sched")
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
    if model_name == 'opt':
        model_cards = opt.model_cards
        assert model_size in model_cards, f"model size {model_size} is not in model cards {model_cards.keys()}"
        config = model_cards[model_size]
        loaded_llm_cpu = OPTForCausalLMSeq._from_config(config, torch_dtype=torch.float16)
    elif model_name == 'bloom':
        model_cards = bloom.model_cards
        assert model_size in model_cards, f"model size {model_size} is not in model cards {model_cards.keys()}"
        config = model_cards[model_size]
        loaded_llm_cpu = BloomForCausalLMSeq._from_config(config, torch_dtype=torch.float16)

    # load the fake calibration data
    caliber = lptorch.inner_caliber
    caliber.set_fake() 
    caliber.load_fake_calib_data(f'fake_calib_{model_name}_{model_size}.pkl')


    method = args.method
    loaded_llm_cpu.eval() # eval mode
    
    # control the token generation
    num_tokens_to_generate = args.num_tokens_to_generate
    prompt_length = args.prompt_length
    bs_token = args.bs_token # how many sentence in a batch
    request_numbers = args.request_numbers # how many requests


    pipeline_strategy_result_qpipe = f"{method}_{model_size}_Tesla_T4_2_Tesla_V100-SXM2-32GB_1_final_strategy.pkl"
    pipeline_strategy_result_qpipe = f'/opt/tiger/launch/qsync/QPipe/scripts/part_strategy/{pipeline_strategy_result_qpipe}'
    sharding_strategy = pickle.load(open(pipeline_strategy_result_qpipe, "rb"))
    loaded_llm_cpu._verify_shard_strategy(sharding_strategy)

    infer_configs = (bs_token, prompt_length, num_tokens_to_generate, request_numbers)

    # init env
    seed = 42
    init_random_seed(seed)
    dist_cfg = init_env()
    assert dist_cfg.world_size > 1, "world size should be larger than 1, else single device"

    if dist_cfg.rank == 0:
        model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
        model_pre_and_post = model_pre_and_post.cuda()

    run_pipeline_p2p(loaded_llm_cpu, dist_cfg, sharding_strategy=sharding_strategy)
