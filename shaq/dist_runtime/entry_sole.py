
from qllm.utils import batch_encode_plus, get_model_size_cuda,  to_device_recursive, greedy_processor
import lptorch
import torch
import copy
import os 
from shaq import (
    init_random_seed,
    fetch_prompts
)
import pickle
from qllm.tp import utils as tp_utils
from qllm.utils.argparser import model_sample_gen_argparser
import qllm.utils as qllm_utils
from qllm.models import qllm_load_pretrained_from_size, init_tokenizer,  create_empty_model
from qllm.models import qllm_load_pretrained_from_size, get_decoder_layer_nums
from qllm.scheduler import DSScheduler
from time import perf_counter
# import sharding strategies helper
from utils import create_uniform_sharding_strategies, parse_args
from shaq.logger import logger
final_result = {}
request_input_ids = {}
request_logit_processor = {}
request_loop_counter = {}
request_unfinished_sequences = {}

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

if __name__ == '__main__':
    args = parse_args()
    bitwidth = args.bitwidth
    if type(bitwidth) is not int and bitwidth.isnumeric():
        args.bitwidth = int(bitwidth)
    
    if args.sample_run:
        # uniform partition and uniform bitwidth samples
        # but also not use the hybrid batch size
        num_shards = args.num_shards
        prefill_bs = args.bs_token // num_shards
        decoder_bss = [args.bs_token // num_shards] * num_shards
    
        bs_token = args.bs_token
        # set by default
        num_tokens_to_generate = args.num_tokens_to_generate
        max_tokens_to_generate = args.max_tokens_to_generate
        prompt_length = args.prompt_length
        # sharing strategy
        bitwidth = args.bitwidth
        decoder_layer_nums = get_decoder_layer_nums(args.model_name, args.model_size)
        sharding_strategy = create_uniform_sharding_strategies(num_shards, decoder_layer_nums, bitwidth)
    else:
        # load the strategy generated by the SHAQ
        method = args.method
        sol_file = f"{args.strat_file_name}.pkl"
        root_dir = os.environ['ROOT_DIR']
        strat_folder = f'{root_dir}/scripts/part_strategy'
        sols_path = f'{strat_folder}/{sol_file}'
        sols = pickle.load(open(sols_path, "rb"))
        # handle the old version
        if 'model_name' in sols:
            model_name = sols['model_name']
            model_size = sols['model_size']
            args.model_name = model_name
            args.model_size = model_size
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

    # test case
    model_name = args.model_name
    model_size = args.model_size
    '''
        Performance mode
        - when you only concerns the performance rather than accuracy
    '''
    # set env
    tokenizer, key = init_tokenizer(model_name, model_size)
    if args.perf_mode:
        os.environ['SET_DECODERS_META'] = "1"
        os.environ['PERF_MODE'] = "1"
        loaded_llm_cpu = create_empty_model(model_name, model_size)
    else:
        load_in_np = os.environ.get('LOAD_IN_NP', '0') == '1'
        if load_in_np:
            # specify the weight folder
            os.environ['NP_WEIGHT_FOLDER'] = os.environ.get('NP_WEIGHT_FOLDER', '/data/llms/converted_weights_np') + f"/{model_name}_{model_size}"
            # load model
            os.environ['SET_DECODERS_META'] = "1"
            loaded_llm_cpu = create_empty_model(model_name, model_size)
            qllm_utils.load_np_weight_opt_non_layer(os.environ['NP_WEIGHT_FOLDER'], loaded_llm_cpu)
        else:
            # case when CPU memory is abundant, direcly load the converted weight
            data_llms_folder = os.environ.get('TRANSFORMERS_CACHE', '/data/llms')
            target_storage_folder = f'{data_llms_folder}/converted_weights'
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
    file_path = os.path.abspath(__file__) if os.environ.get('CALIB_ROOT_FOLDER', None) is None else os.environ.get('CALIB_ROOT_FOLDER')
    calib_file_name = f'fake_calib_{model_name}_{model_size}.pkl'
    calib_file_path = os.path.join(file_path, calib_file_name)
    caliber.load_fake_calib_data(calib_file_path)

    # warmup for performance
    warmup_tokens = args.warmup_tokens
    # hybrid micro-batch scheduler
    chunk_size = len(decoder_bss)
    ds_scheduler = DSScheduler(prefill_bs, decoder_bss)
    # print(prefill_bs, decoder_bss)
    # configs
    infer_configs = (bs_token, prompt_length, num_tokens_to_generate, chunk_size)
    loaded_llm_cpu._verify_shard_strategy(sharding_strategy)

    # init env
    seed = args.seed
    init_random_seed(seed)

    # shard models
    model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
    model_shard_nums = len(sharding_strategy.keys())
    model_packs = [loaded_llm_cpu.shard_model(sharding_strategy, i) for i in range(model_shard_nums)]
    # check whether the packs number equals to the sharding strategy
    assert len(model_packs) == len(sharding_strategy), "model packs number should be equal to the sharding strategy"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Shard on the same gpu for reference

    data_chunks = []
    pad_token_id = None 
    sample_prompts = fetch_prompts(bs_token, prompt_length)
    sample_prompts_dict = ds_scheduler.split_list_of_prompts(sample_prompts)
    prefill_bs_indexes = ds_scheduler.create_ds_indexes()
    for request_id, sample_prompts_list in sample_prompts_dict.items():
        sample_prompts_dict[request_id] = to_device_recursive( \
            [dict(batch_encode_plus(tokenizer, text, return_tensors="pt", max_length=prompt_length)) for text \
                        in sample_prompts_list], device)
    # move all models to cuda
    model_pre_and_post = model_pre_and_post.cuda()
    [model.decoder_layers_to_device(device) for model in model_packs]
    # move tensor to device
    batched_ids = to_device_recursive(dict(data_chunks), device)

    # print model 1, 2, 3 size in MB
    for idx, model in enumerate(model_packs):
        print("Model {} size: ".format(idx), get_model_size_cuda(model, 'MB'))
    
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
            request_token = to_device_recursive(request_token, 'cuda')
            data_chunks.append(request_token)
    # set global vars.
    for chunk_idx, input_id in enumerate(input_id_dict.values()):
        set_input_ids_globals(chunk_idx, input_id)
    
    # be careful that, always shards before quantization.
    # some quantizer like llm.int8 triggers when the model is run cuda() or to(device)
    # if you first move the model to cuda, then shard it, the quantizer will not work

    def generate_one_token(request_token):
        with torch.no_grad():
            intermediate_results = request_token
            for model in model_packs:
                intermediate_results = model.decode(intermediate_results)

        if len(intermediate_results) <= 1:
            return
        request_id = intermediate_results[-2]
        if isinstance(request_id, torch.Tensor):
            request_id = request_id.item()
        # preprocessing  
        outputs = model_pre_and_post.postprocess(intermediate_results, None)
        # 2361 - 2385
        next_token_logits = outputs.logits[:, -1, :]
        logits_processor = request_logit_processor[request_id]
        input_ids = request_input_ids[request_id]
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        # next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        attention_mask = intermediate_results[1]
        flag, concat_tokens, attention_mask = ds_scheduler.pass_scheduler(request_id, next_tokens, attention_mask)
        if flag:
            new_input_ids = torch.cat([input_ids, concat_tokens], dim=-1)
            request_input_ids[request_id] = new_input_ids # record
            # new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            request_token = model_pre_and_post.preprocess_one_token(new_input_ids, concat_tokens, attention_mask=attention_mask, use_cache=True, request_id=request_id)
            logger.info(f"Request id {request_id} done for token {len(new_input_ids)}")
            return request_token
        else:
            return None, None

    assert len(model_packs) == 1, "one card one model"
    for chunk_id in range(chunk_size):
        # init kv cache for each decoder bs
        # the prefill stage will use the same cache created by decoder
        model_packs[0].init_kv_cache(decoder_bss[chunk_id], prompt_length, max_tokens_to_generate, chunk_id)
    model_packs[0].eval()
    module = model_packs[0]

    # warmup 
    # prefill
    def gen_n_tokens(n):
        for request_token in data_chunks:
            request_token = generate_one_token(request_token)
        # decocode
        for i in range(n - 1):
            request_token = generate_one_token(request_token)
    
    gen_n_tokens(2) # warmup
    module._reset_kv_status()
    ds_scheduler.reset_status()
    for chunk_idx, input_id in enumerate(input_id_dict.values()):
        set_input_ids_globals(chunk_idx, input_id)
    tik_data = perf_counter()
    gen_n_tokens(num_tokens_to_generate)
    tok_data = perf_counter()
    latency = tok_data - tik_data
    throughput = bs_token / latency
    token_throughput = throughput * num_tokens_to_generate
    logger.info("Latency is %f, throughput is %f", latency, throughput)
    logger.info('Token throughput is %f', token_throughput)

    for request_id, result in request_input_ids.items():
        generated_text = tokenizer.batch_decode(result, skip_special_tokens=True)
        print("request_id: {}, generated_text: {}".format(request_id, generated_text))

    


    



