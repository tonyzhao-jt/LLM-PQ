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