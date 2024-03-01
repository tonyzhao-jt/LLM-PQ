import argparse
import os 
import torch
import time
import pandas as pd


from transformers import LogitsProcessorList
from transformers import (
    AutoTokenizer
)


from qllm.models import create_empty_model, create_model_config, return_h1_h2
import qllm.utils as qllm_utils

from llm_pq.logger import init_logger
logger = init_logger(__name__)
from llm_pq.utils import get_device_name_and_mem
from llm_pq import (
    fetch_prompts
)

def parse_args():
    parser = argparse.ArgumentParser(description='Profile a transformer model')
    parser.add_argument('--model-name', type=str, default='opt', help='model name')
    parser.add_argument('--model-size', type=str, default='175b', help='Size of the transformer model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size range(1,32)')
    parser.add_argument('--prompt_length', type=int, default=1, help='Length of input sequence')
    parser.add_argument('--step', type=int, default=10, help='Profiled step')
    parser.add_argument('--repeat', type=int, default=100, help='Number of iterations to profile')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--bit', type=str, default='8:tc', help='Precision bit setting')
    args = parser.parse_args()
    return args


def handle_results(final_intermediate_result, input_ids, logits_processor) -> None:
    request_id = final_intermediate_result[-2]
    if isinstance(request_id, torch.Tensor):
        request_id = request_id.item()
    outputs = model_pre_and_post.postprocess(final_intermediate_result, None)
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    if len(next_tokens.shape) == 1:
        next_tokens = next_tokens.view(-1,1)
    new_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
    # new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    request_token = model_pre_and_post.preprocess_one_token(new_input_ids, next_tokens, use_cache=True, request_id=request_id)

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
    return logits_processor

    
def init_tokenizer(model_name):
    if model_name == 'opt':
        return AutoTokenizer.from_pretrained("facebook/opt-66b")
    elif model_name == 'bloom':
        return AutoTokenizer.from_pretrained("bigscience/bloom")
    
if __name__ == '__main__':
    # set env
    os.environ['SET_DECODERS_META'] = "1"
    os.environ['PERF_MODE'] = "1"
    args = parse_args()
    # test case
    model_name = args.model_name
    model_size = args.model_size
    batch_size = args.batch_size
    prompt_length = args.prompt_length
    loaded_llm_cpu = create_empty_model(model_name, model_size)
    tokenizer = init_tokenizer(model_name)
    config = create_model_config(model_name, model_size)
    h1, h2 = return_h1_h2(config)
    device_name, device_mem, _ = get_device_name_and_mem()
    step = args.step

    loaded_llm_cpu.eval()
    loaded_llm_cpu.cuda()

    # generate input_ids with the next tokens
    # two cases 
    # 1. prefill stage, the intermediate should equal to the prompt length
    # 2. decode stage, seq = 1, past seq length = s + i
    repeat = args.repeat
    warmup = args.warmup

    request_id = 0
    model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
    model_pre_and_post = model_pre_and_post.cuda()
    generation_config = model_pre_and_post.generation_config

    folder_name = "lat_prepost_profiled_result"
    # make a folder under the current directory if not exists
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


    def run_prepose_profile(p_batch_size, p_prompt_length, prefill=False):
        device = torch.device("cuda:0")
        sample_texts = fetch_prompts(batch_size, prompt_length)
        input_ids = dict(tokenizer.batch_encode_plus(sample_texts, padding='max_length', max_length=prompt_length, return_tensors="pt"))
        input_ids = qllm_utils.to_device_recursive(input_ids, device=device)
        input_ids = input_ids['input_ids']
        logits_processor = set_input_ids_globals(request_id, input_ids)
        # for prefill
        if prefill:
            fake_final_intemediate = torch.rand([p_batch_size, p_prompt_length, h1]).cuda().half()
        else:
            # for decode
            fake_final_intemediate = torch.rand([p_batch_size, 1, h1]).cuda().half()

        # construct input for 
        if model_name == 'opt':
            # 6 entries
            final_intermediate = [fake_final_intemediate, None, None, None, None, request_id, None]
        
        if model_name == 'bloom':
            # 7 entries
            final_intermediate = [fake_final_intemediate, None, None, None, None, None, request_id, None]

        device = torch.device("cuda:0")
        model_pre_and_post.return_dict = True
        final_intermediate = qllm_utils.object_to_tensor(final_intermediate)
        final_intermediate = qllm_utils.to_device_recursive(final_intermediate, device=device)
        final_intermediate = qllm_utils.to_dtype_recursive(final_intermediate, dtype=torch.float16)

        for _ in range(warmup):
            res = handle_results(final_intermediate, input_ids, logits_processor)
        start = time.time()
        torch.cuda.synchronize()
        for _ in range(repeat):
            res = handle_results(final_intermediate, input_ids, logits_processor)
        torch.cuda.synchronize()
        end = time.time()
        return (end - start) / repeat * 1000
    

    # pandas columns
    columns = ['model_name', 'model_size', 'h1', 'h2', 'batch_size', 'prompt_length', 'stage', 'time']
    csv_file_name = f'{device_name}_{model_size}_prepose.csv'
    csv_file_name = os.path.join(folder_name, csv_file_name)
    if os.path.exists(csv_file_name):
        df = pd.read_csv(csv_file_name)
    else:
        df = pd.DataFrame(columns=columns)
    for stage in [0, 1]: # prefill or not
        for prompt_length in [256]:
            for batch_size in [1, 2, 3, 4, 5, 6, 8]:
                # check whether entry has been profiled
                if len(df[(df['batch_size'] == batch_size) & (df['prompt_length'] == prompt_length) & (df['stage'] == stage)]) > 0:
                    logger.info(f'batch_size: {batch_size}, prompt_length: {prompt_length}, stage: {stage} has been profiled')
                    continue
                lat = run_prepose_profile(batch_size, prompt_length, prefill=stage==0)
                logger.info(f'batch_size: {batch_size}, prompt_length: {prompt_length}, stage: {stage}, latency: {lat}')
                if pd.__version__ > '1.3.5':
                    df = df._append({
                        'model_name': model_name,
                        'model_size': model_size,
                        'h1': h1,
                        'h2': h2,
                        'batch_size': batch_size,
                        'prompt_length': prompt_length,
                        'stage': stage,
                        'time': lat
                    }, ignore_index=True)
                else:
                    df = df.append({
                        'model_name': model_name,
                        'model_size': model_size,
                        'h1': h1,
                        'h2': h2,
                        'batch_size': batch_size,
                        'prompt_length': prompt_length,
                        'stage': stage,
                        'time': lat
                    }, ignore_index=True)
            df.to_csv(csv_file_name, index=False)

    # decode
    for stage in [1]: # prefill or not
        for prompt_length in [256]:
            for past_seq_length_i in range(10, 110, step):
                past_seq_length = prompt_length + past_seq_length_i
                for batch_size in [1, 2, 3, 4, 5, 6, 8]:
                    # check whether entry has been profiled
                    if len(df[(df['batch_size'] == batch_size) & (df['prompt_length'] == past_seq_length) & (df['stage'] == stage)]) > 0:
                        logger.info(f'batch_size: {batch_size}, past_seq_length: {past_seq_length}, stage: {stage} has been profiled')
                        continue
                    lat = run_prepose_profile(batch_size, past_seq_length, prefill=stage==0)
                    logger.info(f'batch_size: {batch_size}, past_seq_length: {past_seq_length}, stage: {stage}, latency: {lat}')
                    if pd.__version__ > '1.3.5':
                        df = df._append({
                            'model_name': model_name,
                            'model_size': model_size,
                            'h1': h1,
                            'h2': h2,
                            'batch_size': batch_size,
                            'prompt_length': past_seq_length,
                            'stage': stage,
                            'time': lat
                        }, ignore_index=True)
                    else:
                        df = df.append({
                            'model_name': model_name,
                            'model_size': model_size,
                            'h1': h1,
                            'h2': h2,
                            'batch_size': batch_size,
                            'prompt_length': past_seq_length,
                            'stage': stage,
                            'time': lat
                        }, ignore_index=True)
                df.to_csv(csv_file_name, index=False)
    