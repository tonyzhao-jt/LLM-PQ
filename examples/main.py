

from transformers import AutoTokenizer

# Main 3rd Party
import torch 
import torch.distributed as dist
import os
import pickle
import time
from time import perf_counter
# lptorch related
import lptorch 

# QLLM related
from qllm.models import create_empty_model
from qllm.scheduler import DSScheduler
from qllm.utils import batch_encode_plus, get_model_size_cuda,  to_device_recursive, greedy_processor
from qllm.models import bloom, opt, qllm_load_pretrained_from_size

# QPipe Related
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

# example utils
from example_utils import create_uniform_sharding_strategies


master_stage_context = None
lock_queue = None
work_queue = None
simple_queue_thread = None
results_counter = ThreadSafeCounter()
final_result = {}
request_input_ids = {}
request_logit_processor = {}
request_loop_counter = {}
request_unfinished_sequences = {}
pad_token_id = None
def handle_results(final_intermediate_result) -> None:
    if len(final_intermediate_result) <= 1:
        return
    request_id = final_intermediate_result[-2]
    if isinstance(request_id, torch.Tensor):
        request_id = request_id.item()
    results_counter.add(1)
    # print(results_counter._value)
    # get original input id
    input_ids = request_input_ids[request_id]
    logits_processor = request_logit_processor[request_id]
    unfinished_sequences = request_unfinished_sequences[request_id]
    # generate new tokens
    final_intermediate_result = to_device_recursive(final_intermediate_result, 'cuda:0')
    outputs = model_pre_and_post.postprocess(final_intermediate_result, None)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
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
            # print(request_token)
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
    global pad_token_id
    logits_processor, (unfinished_sequences, p_pad_token_id) = greedy_processor(loaded_llm_cpu, p_input_ids, max_tokens_to_generate, prompt_length)
    if pad_token_id is None:
        pad_token_id = p_pad_token_id
    final_result[request_id] = None
    request_input_ids[request_id] = p_input_ids
    request_logit_processor[request_id] = logits_processor
    request_loop_counter[request_id] = 0
    request_unfinished_sequences[request_id] = unfinished_sequences


from datetime import timedelta


def init_tokenizer(model_name):
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
    global prefill_cnt, simple_queue_thread, lock_queue, work_queue
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

    if rank == data_rank:
        device = torch.device(f'cuda:{local_rank}')

        data_chunks = []
        sample_prompts = fetch_prompts(bs_token, prompt_length)
        sample_prompts_dict = ds_scheduler.split_list_of_prompts(sample_prompts)
        prefill_bs_indexes = ds_scheduler.create_ds_indexes()
        for request_id, sample_prompts_list in sample_prompts_dict.items():
            sample_prompts_dict[request_id] = to_device_recursive( \
                [dict(batch_encode_plus(tokenizer, text, return_tensors="pt", max_length=prompt_length)) for text \
                            in sample_prompts_list], device)
        data_chunks = []
        input_id_dict = {}
        for request_id, prefill_bs_indexes in prefill_bs_indexes.items():
            for idx, prefill_bs_index in enumerate(prefill_bs_indexes):
                current_sub_request_batch_ids = sample_prompts_dict[request_id][idx]
                if request_id not in input_id_dict:
                    input_id_dict[request_id] = current_sub_request_batch_ids['input_ids']
                else:
                    input_id_dict[request_id] = torch.cat([input_id_dict[request_id], current_sub_request_batch_ids['input_ids']], dim=0)
                # print(current_sub_request_batch_ids['input_ids'].shape)
                request_token = model_pre_and_post.preprocess(**current_sub_request_batch_ids, use_cache=True, request_id=request_id, batch_index=prefill_bs_index)
                if not args.nccl:
                    request_token = to_device_recursive(request_token, 'cpu')
                data_chunks.append(request_token)
        # set global vars.
        for chunk_idx, input_id in enumerate(input_id_dict.values()):
            set_input_ids_globals(chunk_idx, input_id)
        prefill_cnt = len(data_chunks)
        print("Pipeline Data Loaded, with prefill cnts: ", prefill_cnt)
        # for i in range(prefill_cnt):
        #     print(data_chunks[i][0].shape)
    
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

    with DistP2pContext(('gloo',), { 'world_size': world_size, 'rank': rank, 'timeout': timedelta(seconds=3000)}, handle_cmd) \
        as dist_ctx:
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
                time.sleep(10)
                module._reset_kv_status()
                dist.barrier()
                run_inf(stage_ctx, input_id_dict, data_chunks, sample_num=original_num_tokens_to_generate)
                dist_ctx.cmd_broadcast(CMD_STOP)
                # print the result
                if not args.perf_mode:
                    for request_id, result in request_input_ids.items():
                        generated_text = tokenizer.batch_decode(result, skip_special_tokens=True)
                        print("request_id: {}, generated_text: {}".format(request_id, generated_text))
                # join the queue thread
                stop_event.set()
            else:
                dist.barrier()
                time.sleep(10)
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
    parser.add_argument("--nccl", action='store_true', default=False, help="use nccl")
    parser.add_argument("--warmup_tokens", type=int, default=2, help="warmup")
    parser.add_argument("--method", type=str, default="adaqpipe", help="method of sched")
    parser.add_argument("--strat_file_name", type=str, default=None)
    # random seed
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # enable by sample flag
    parser.add_argument("--sample-run", action='store_true', default=False, help="sample run")
    # add bitwidth, shard_number
    parser.add_argument('--bitwidth', default=16)
    parser.add_argument('--num-shards', type=int, default=2) # 2 cards by default
    # perfmode
    parser.add_argument('--perf-mode', action='store_true', default=False)

    parser.parse_args()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    # test case
    model_name = args.model_name
    model_size = args.model_size
    '''
        Performance mode
        - when you only concerns the performance rather than accuracy
    '''
    # set env
    if args.perf_mode:
        os.environ['SET_DECODERS_META'] = "1"
        os.environ['PERF_MODE'] = "0"
        loaded_llm_cpu = create_empty_model(model_name, model_size)
        # init tokenizer
        tokenizer = init_tokenizer(model_name)
    else:
        # load weight
        target_storage_folder = '/data/llms/converted_weights'
        # load model 
        if model_name == 'bloom':
            qllm_model, tokenizer, key = qllm_load_pretrained_from_size(model_name, model_size)
        elif model_name == 'opt':
            # use converted weight
            path = os.path.join(target_storage_folder, f"{model_name}_{model_size}")
            if not os.path.exists(path):
                raise ValueError("Please run weight_convert.py first")
            qllm_model, tokenizer, key = qllm_load_pretrained_from_size(model_name, model_size, target_storage_folder=target_storage_folder)
        loaded_llm_cpu = qllm_model

    # load the fake calibration data
    caliber = lptorch.inner_caliber
    caliber.set_fake() 
    file_path = os.path.abspath(__file__)
    calib_file_name = f'fake_calib_{model_name}_{model_size}.pkl'
    calib_file_path = os.path.join(os.path.dirname(file_path), 'fake_calib', calib_file_name)
    caliber.load_fake_calib_data(calib_file_path)

    if args.sample_run:
        # uniform partition and uniform bitwidth samples
        # but also not use the hybrid batch size
        prefill_bs = args.bs_token 
        decoder_bss = [args.bs_token] 
        bs_token = args.bs_token
        # set by default
        num_tokens_to_generate = args.num_tokens_to_generate
        max_tokens_to_generate = args.max_tokens_to_generate
        prompt_length = args.prompt_length
        # sharing strategy
        num_shards = args.num_shards
        bitwidth = args.bitwidth
        decoder_layer_nums = loaded_llm_cpu.get_decoder_layer_num()
        sharding_strategy = create_uniform_sharding_strategies(num_shards, decoder_layer_nums, bitwidth)
    else:
        # load the strategy generated by the SHAQ
        method = args.method
        sol_file = f"{args.strat_file_name}.pkl"
        strat_folder = '/workspace/qpipe/scripts/part_strategy'
        sols_path = f'{strat_folder}/{sol_file}'
        sols = pickle.load(open(sols_path, "rb"))
        num_tokens_to_generate = sols['mu_n']
        max_tokens_to_generate = sols['n']
        bs_token = sols['gloabl_bz'] # how many sentence in a batch
        assert args.method in sols, f"no {args.method} in {sols_path}"
        # get sols info
        sol = sols[args.method]
        sharding_strategy = sol['use_plan']
        prefill_bs = sol['prefill_bz']
        decoder_bss = sol['bz_decode_bss']
        prompt_length = args.prompt_length if sols.get('prompt_length') is None else sols['prompt_length']

    # warmup for performance
    warmup_tokens = args.warmup_tokens
    # hybrid micro-batch scheduler
    chunk_size = len(decoder_bss)
    ds_scheduler = DSScheduler(prefill_bs, decoder_bss)
    # configs
    infer_configs = (bs_token, prompt_length, num_tokens_to_generate, chunk_size)
    loaded_llm_cpu._verify_shard_strategy(sharding_strategy)

    # init env
    seed = args.seed
    init_random_seed(seed)
    dist_cfg, hard_device_mesh = init_env()
    # assert dist_cfg.world_size > 1, "world size should be larger than 1, else single device"

    if dist_cfg.rank == 0:
        model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
        model_pre_and_post = model_pre_and_post.cuda()

    run_pipeline_p2p(loaded_llm_cpu, dist_cfg, sharding_strategy=sharding_strategy)
    if simple_queue_thread is not None:
        simple_queue_thread.join()
    





