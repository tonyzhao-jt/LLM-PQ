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