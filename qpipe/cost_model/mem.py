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

