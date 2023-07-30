# sharding_strategy = {
#     0: {
#     },
#     1: {
#         0: {'shard': [0, 1], 'bits': [16, 16]},
#         1: {'shard': [0, 1], 'bits': [16, 16]},
#         2: {'shard': [0, 1], 'bits': [16, 16]},
#         3: {'shard': [0, 1], 'bits': [16, 16]},
#         4: {'shard': [0, 1], 'bits': [16, 16]},
#         5: {'shard': [0, 1], 'bits': [16, 16]},
#         6: {'shard': [0, 1], 'bits': [16, 16]},
#         7: {'shard': [0, 1], 'bits': [16, 16]},
#         8: {'shard': [0], 'bits': [16]},
#     },
#     2: {
#         8: {'shard': [1], 'bits': [16]},
#         9: {'shard': [0,1], 'bits': [16, 16]},
#         10: {'shard': [0,1], 'bits': [16, 16]},
#         11: {'shard': [0,1], 'bits': [16, 16]},
#         # 350M
#         12: {'shard': [0,1], 'bits': [16, 16]},
#         13: {'shard': [0,1], 'bits': [16, 16]},
#         14: {'shard': [0,1], 'bits': [16, 16]},
#     },
#     3:{
#         15: {'shard': [0,1], 'bits': [16, 16]},
#         16: {'shard': [0,1], 'bits': [16, 16]},
#         17: {'shard': [0,1], 'bits': [16, 16]},
#         18: {'shard': [0,1], 'bits': [16, 16]},
#         19: {'shard': [0,1], 'bits': [16, 16]},
#         20: {'shard': [0,1], 'bits': [16, 16]},
#         21: {'shard': [0,1], 'bits': [16, 16]},
#         22: {'shard': [0,1], 'bits': [16, 16]}, 
#         23: {'shard': [0,1], 'bits': [16, 16]},
#     }
# }  

from qllm.utils import partition_a_into_b_bins

def create_uniform_sharding_strategies(shards_num, decoder_layer_nums, bitwidth):
    sharding_strategy = {}
    each_layer_shards = partition_a_into_b_bins(decoder_layer_nums, shards_num) # e.g. [8,8,8]
    decoder_layer_range_for_each_shard = []
    for i in range(shards_num):
        decoder_layer_range_for_each_shard.append((sum(each_layer_shards[:i]), sum(each_layer_shards[:i+1])))
    
    for shard in range(shards_num):
        sharding_strategy[shard] = {}
        shard_decoders = decoder_layer_range_for_each_shard[shard]
        for layer in range(shard_decoders[0], shard_decoders[1]):
            sharding_strategy[shard][layer] = {'shard': [0, 1], 'bits': [bitwidth] * 2}
    return sharding_strategy

# create_uniform_sharding_strategies(3, 24, 16)


# parser
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
    parser.add_argument('--no_auto', action='store_true', default=False)
    # workload test
    parser.add_argument('--workload-test', action='store_true', default=False)
    parser.add_argument('--workload-nums', type=int, default=10)
    parser.add_argument('--sampler-lower', type=float, default=0.6) # default sampler lower bound U(0.6, 1)

    parser.parse_args()
    args = parser.parse_args()
    return args