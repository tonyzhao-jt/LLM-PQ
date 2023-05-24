import argparse
import os 
from qllm.models import create_empty_model
import torch
import time
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
    parser.parse_args()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # set env
    # os.environ['SET_DECODERS_META'] = "1"
    # os.environ['PERF_MODE'] = "1"
    # args = parse_args()
    # # test case
    # model_name = args.model_name
    # model_size = args.model_size
    # loaded_llm_cpu = create_empty_model(model_name, model_size)

    # loaded_llm_cpu.eval()
    # loaded_llm_cpu.cuda()

    # all token size
    # all_token_size = []
    # # create token concat
    # for i in range(args.num_tokens_togenerate):
    #     token_size = [args.bs_token, i + args.prompt_length]
    
    bs = 8
    token_size = [1, 1]
    token_prev = torch.rand([bs, 512]).cuda()
    cnt = 10
    # create token
    tokens = [torch.rand(token_size).cuda() for i in range(bs)]
    # concate
    start = time.time()
    torch.cuda.synchronize()
    for _ in range(cnt):
        new_tokens = torch.cat(tokens, dim=0)
    torch.cuda.synchronize()
    end = time.time()
    print(end-start)

    # # create token
    start = time.time()
    torch.cuda.synchronize()
    for _ in range(cnt):
        tokens = torch.cat([new_tokens, token_prev], dim=-1)
    torch.cuda.synchronize()
    end = time.time()
    print(end-start)