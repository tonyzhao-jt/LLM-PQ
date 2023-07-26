from ..partitioner.helper import (
    get_single_device_mem_constraints,
)
from ..cost_model import (
    get_mem_with_layer_bit_pair
)
import numpy as np
def get_M_with_bitwidth_pair(BITs, model_mem_estimator, group_L, group_size):
    mem_bits_vector = get_mem_with_layer_bit_pair(BITs, model_mem_estimator)
    M = np.tile(mem_bits_vector, (group_L, 1)) * group_size # repeat the mem_bits_vector for group_L times
    M = np.ceil(M).astype(int) # ceil
    return M

def get_device_topo_available_mem_with_order(current_D, model_mem_estimator, prefill_bz, bz_decode_max, time_mult_times=1):
    M_d = np.array([get_single_device_mem_constraints(device_name) for d_rank, device_name in current_D.items()]) 
    # reduce the embedding size on device 0
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size_with_bz(prefill_bz, bz_decode_max, unit='MB')[0] 
    temp_later_decode = model_mem_estimator.calculate_temp_tensor_size_next_i(unit='MB')[0]
    M_d[0] -= post_pre_mem
    if len(M_d) > 1:
        M_d[1:] -= temp_later_decode * time_mult_times
    M_d[0] -= max(temp_tensor_mem, temp_later_decode * time_mult_times)
    return M_d

from ..cost_model import (
    estimate_single_layer_mem,
)
def estimate_single_device_mem(layers_range, bit_assignment, model_mem_estimator):
    i, j = layers_range
    i_to_j_mem = 0
    # k % 2 means shard
    for k in range(i, j):
        layer_x_mem = estimate_single_layer_mem(model_mem_estimator, 0, bit_assignment[k * 2]) + \
                        estimate_single_layer_mem(model_mem_estimator, 1, bit_assignment[k * 2 + 1])
        # print("layer_x_mem", layer_x_mem)
        i_to_j_mem += layer_x_mem
    return i_to_j_mem


# first make sure the partition is within the memory budget
def check_memory_budget_single_device(device_mem, device_rank, layers_range, bit_assignment, model_mem_estimator):
    i_to_j_mem = estimate_single_device_mem(layers_range, bit_assignment, model_mem_estimator)
    if i_to_j_mem > device_mem:
        print(f"memory budget exceeded for device {device_rank}, {i_to_j_mem} > {device_mem}")
        return False
    return True

def check_memory_budget(res, model_mem_estimator, name='shaq'):
    plan = res['plan']
    partition_result = plan['partition_result']
    bit_assignment = plan['bit_assignment']
    D = res['D']
    prefill_bz = res['prefill_bz']
    bz_decode_max = res['bz_decode_max']
    bs_pack = (prefill_bz, bz_decode_max)
    # print("verify memory budget for", name)
    D_mem = get_device_topo_available_mem_with_order(D, model_mem_estimator, prefill_bz, bz_decode_max)
    for device_rank, layers_range in partition_result.items():
        device_mem = D_mem[device_rank]
        flag = check_memory_budget_single_device(device_mem, device_rank, layers_range, bit_assignment, \
                                           model_mem_estimator)
        if not flag:
            print("memory budget exceeded, return False", name)
            import pdb; pdb.set_trace()
            return False
    # print("all passed")
    return True