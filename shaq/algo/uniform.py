from .. import _globals
from ..partitioner.helper import (
    init_parameters_and_cost_models, 
    create_device_mesh_and_mem,
    get_single_device_mem_constraints,
    get_device_info,
)

from ..partitioner.utils import (
    assign_uniform_bit
)

from ..cost_model import (
    estimate_all_layer_mem
)

from ..utils import (
    partition_a_into_b_bins,
    get_default_decode_bz
)

from .utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE,
    convert_to_shaq_result2partitions
)

# default libs
import pickle
import os 
import numpy as np 

from .lat_utils import run_simu

# setup ilp configs
unit = _globals.MEM_UNIT

from ..utils import (
    get_default_decode_bz,
    get_factors
)

def generate_uniform_partition(model_mem_estimator, T, max_device_mem, num_devices, num_hidden_layers, D, bz_pack, bit=8):
    time_mult_times = _globals.TIME_MULT_TIMES
    (global_bz, prefill_bz, bz_decode_max)= bz_pack 
    bit_assignment = {}
    assign_uniform_bit(T, bit, bit_assignment)
    mem_required = estimate_all_layer_mem(model_mem_estimator, T, bit_assignment)
    if mem_required > max_device_mem:
        print(f"Total memory required: {mem_required / 1024 } GB", "available memory: ", max_device_mem / 1024, "GB")
        print("The model is too large to fit in the device mesh")
        return NOT_AVAILABLE
    # perform uniform partition.
    each_device_mem_availability = [get_single_device_mem_constraints(device_name) for d_rank, device_name in D.items()]
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size_with_bz(prefill_bz, bz_decode_max, unit='MB')[0] 
    temp_later_decode = model_mem_estimator.calculate_temp_tensor_size_next_i(unit='MB')[0]
    each_device_mem_availability[0] -= (post_pre_mem + max(temp_tensor_mem, temp_later_decode * time_mult_times))
    # layer partitioned memory
    each_device_mem = mem_required // num_devices
    if each_device_mem > min(each_device_mem_availability):
        print("The model is too large to fit in the device mesh")
        print("min(each_device_mem_availability): ", min(each_device_mem_availability), "Uniform requries mem: ", each_device_mem)
        return NOT_AVAILABLE
    # create the partition
    # partition T accoding to layer numbers
    allocation = partition_a_into_b_bins(num_hidden_layers, num_devices)
    # allocate
    partition_result = {}
    idx_start = 0
    for d_rank, device_name in D.items():
        layer_nums = allocation[d_rank]
        partition_result[d_rank] = [idx_start, idx_start + layer_nums]
        idx_start += layer_nums
    
    return {
        'partition_result': partition_result,
        'bit_assignment': bit_assignment,
    }


'''
    Initiailization
'''
from ..partitioner import gen_config
# global variables
global_bz, micro_bz = None, None
s, n = None, None
model_size, device_info = None, None
use_profiler_prediction = False
D = None
device_numbers, device_names = None, None
available_bits = None
config = None
theta = None
mu_n = None
ROOT_DIR=os.environ.get('ROOT_DIR', None)
assert ROOT_DIR is not None, "ROOT_DIR is not set"
cost_model_store_path = f'{ROOT_DIR}/scripts/cost_model_store'
comm_cost_model_dir = f'{ROOT_DIR}/scripts/comm_cost_model'
def main(args):
    global global_bz, micro_bz, s, n
    global model_size, device_info
    global use_profiler_prediction
    global D 
    global device_numbers, device_names
    global available_bits
    global config
    global theta
    global mu_n
    global cost_model_store_path, comm_cost_model_dir
    # global variables

    global_bz = gen_config.global_bz
    micro_bz = gen_config.micro_bz
    s = gen_config.s
    n = gen_config.n
    model_size = args.model_size # '66b'
    device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers # [2, 3]
    # generation configs
    config = args.config

    gamma = _globals.gamma # expected generated tokens
    theta = _globals.theta # control the concern for accuracy
    mu_n = int(gamma * n)
    available_bits = _globals.AVAILABLE_BITS # we now can do hardware-aware quantization with 8:tc and 8:tc-li
    D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
    # max_device_mem can be used to check whether OOM or not
    use_profiler_prediction = args.use_profiler_prediction # use regression model to predict or load predictor
    # target model configuration
    device_info = get_device_info(device_names, device_numbers)
    comm_cost_model_dir = f'{args.comm_cost_model_dir}/{device_info}'
    cost_model_store_path = None # initialize the cost model

    if args.init_pack:
        model_mem_estimator, comm_cost_model, lat_cost_model, T = args.init_pack 
    if args.debug:
        model_mem_estimator, comm_cost_model, lat_cost_model, T = init_parameters_and_cost_models(config, device_names, device_numbers, cost_model_store_path, \
                                                                                                        global_bz, micro_bz, s, n, \
                                                                                                    comm_cost_model_folder=comm_cost_model_dir)
        lat_cost_model.update_profiled_result(args.lat_profile_dir)
        if not use_profiler_prediction:
            lat_cost_model.load_regression_cost_model()

    num_hidden_layers = len(T) // 2
    num_devices = len(D)

    # bz_decode_max = get_default_decode_bz(global_bz, num_devices)
    # strat = partition_a_into_b_bins(global_bz, num_devices)
    # prefill_bz = bz_decode_max
    # bz_pack = (global_bz, prefill_bz, bz_decode_max)

    comm_multiplier = args.comm_multiplier
    num_device_all = len(D)
    strat = partition_a_into_b_bins(global_bz, num_device_all)
    bz_decode_max = get_default_decode_bz(global_bz, num_device_all)
    candidate_prefill_bzs = get_factors(bz_decode_max)
    available_bits = _globals.AVAILABLE_BITS_WO_INFO # wo info
    bit = args.uniform_bit


    best_e2e = 1e9
    optimal_sol = None
    # print("uniform bit: ", bit)
    for prefill_bz in candidate_prefill_bzs:
        # print("prefill_bz: ", prefill_bz)
        bz_pack = (global_bz, prefill_bz, bz_decode_max)

        result = generate_uniform_partition(model_mem_estimator, T, max_device_mem, num_devices, num_hidden_layers, D,\
                                            bz_pack, bit)
        res = {}
        if result == NOT_AVAILABLE:
            res['plan'] = NOT_AVAILABLE
            res['obj'] = NOT_AVAILABLE
        else:
            res['plan'] = result 
            res['obj'] = 1
        
        
        best_plan = {
            'prefill_bz': prefill_bz,
            'bz_decode_max': bz_decode_max,
            'bz_decode_bss': strat,
            'device_names': device_names,
            'device_numbers': device_numbers,
            'plan': res['plan'],
            'obj': res['obj'],
            'D': D,
            'maps': None,
            'name': 'uniform',
        }
        if optimal_sol is None:
            optimal_sol = best_plan
        if res['obj'] != NOT_AVAILABLE:
            convert_to_shaq_result2partitions(best_plan)
            e2e_lat = run_simu(gen_config, best_plan, lat_cost_model, comm_cost_model, use_profiler_prediction, mu_n, comm_multiplier)
            if e2e_lat < best_e2e:
                optimal_sol = best_plan
    # for bit in available_bits:
    #     res_bit = generate_uniform_partition(bit)
    #     file_name_bit = f'uniform_partition_bit_{bit}_' + model_size + '_' + device_info + '.pkl'
    #     folder = '/workspace/qpipe/scripts/strategy'
    #     save_with_pickle(res_bit, file_name_bit, folder)
    return optimal_sol


# if __name__ == '__main__':
#     ilp_env()
#     args = common_argparser()
#     args.debug = True
#     res = main(args)
#     print(res)