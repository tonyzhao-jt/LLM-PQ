from .._globals import MEM_UNIT as unit
def estimate_single_layer_mem(estimator, shard, bit):
    partition = {0: {"shard": [shard], "bits": [bit]}}
    mem_require, _ = estimator.calculate_maximum_mem_occupation_of_partition(partition, unit=unit)
    return mem_require

def estimate_all_layer_mem(estimator, layers, bit_map):
    all_mem_require = 0
    for idx, shard in enumerate(layers):
        bit = bit_map[idx]
        mem_require = estimate_single_layer_mem(estimator, shard, bit)
        all_mem_require += mem_require
    return all_mem_require

import numpy as np 
def get_mem_with_layer_bit_pair(bit_pairs, model_mem_estimator): 
    mem_bits_vector = np.zeros(len(bit_pairs))
    for idx, bit_pair in enumerate(bit_pairs):
        attn_bit, ffn_bit = bit_pair
        attn_mem = estimate_single_layer_mem(model_mem_estimator, 0, attn_bit)
        ffn_mem = estimate_single_layer_mem(model_mem_estimator, 1, ffn_bit)
        mem = attn_mem + ffn_mem
        mem_bits_vector[idx] = mem
    return mem_bits_vector