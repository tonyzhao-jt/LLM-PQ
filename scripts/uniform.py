# generate uniform partition for models
# uniform_8: int8 uniform partition
# uniform_16: fp16 uniform partition
from qllm.models.OPT.opt import model_cards
import qpipe
from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    create_device_mesh_and_mem,
    get_single_device_mem_constraints,
    get_device_info,
)



from qpipe.partitioner.utils import (
    assign_uniform_bit
)

from qpipe.cost_model import (
    estimate_all_layer_mem
)

from qpipe.utils import (
    save_with_pickle,
    partition_a_into_b_bins
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, required=True)
parser.add_argument('--device_names',  nargs='+', type=str, required=True)
parser.add_argument('--device_numbers',  nargs='+', type=int, required=True)
args = parser.parse_args()

unit = qpipe._globals.MEM_UNIT
time_mult_times = qpipe._globals.TIME_MULT_TIMES
# model size
model_size = args.model_size # '66b'
device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
device_numbers = args.device_numbers # [2, 3]
assert len(device_names) == len(device_numbers), "device_names and device_numbers should have the same length"


def generate_uniform_partition(bit=8):
    bit_assignment = {}
    assign_uniform_bit(T, bit, bit_assignment)
    mem_required = estimate_all_layer_mem(model_mem_estimator, T, bit_assignment)
    if mem_required > max_device_mem:
        print(f"Total memory required: {mem_required / 1024 } GB", "available memory: ", max_device_mem / 1024, "GB")
        print("The model is too large to fit in the device mesh")
        return None
    # perform uniform partition.
    each_device_mem_availability = [get_single_device_mem_constraints(device_name) for d_rank, device_name in D.items()]
    # layer partitioned memory
    each_device_mem = mem_required // num_devices
    if each_device_mem > min(each_device_mem_availability):
        print("The model is too large to fit in the device mesh")
        print("each_device_mem: ", each_device_mem, "min(each_device_mem_availability): ", min(each_device_mem_availability))
        return None
    # create the partition
    # partition T accoding to layer numbers
    allocation = partition_a_into_b_bins(num_hidden_layers, num_devices)
    # allocate
    partition_result = {}
    idx_start = 0
    for d_rank, device_name in D.items():
        layer_nums = allocation[d_rank] * 2
        partition_result[d_rank] = [idx_start, idx_start + layer_nums]
        idx_start += layer_nums
    
    return {
        'partition_result': partition_result,
        'bit_assignment': bit_assignment,
    }

'''
    Initialization
'''
# help input the config
from qpipe.partitioner import gen_config
# generation configs
global_bz = gen_config.global_bz
micro_bz = gen_config.micro_bz
s = gen_config.s
n = gen_config.n

config = model_cards[model_size]
D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
# max_device_mem can be used to check whether OOM or not
use_profiler_prediction = True
# target model configuration
device_info = get_device_info(device_names, device_numbers)
comm_cost_model_dir = f'/workspace/qpipe/scripts/comm_cost_model/{device_info}'
cost_model_store_path = None # initialize the cost model
model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names, cost_model_store_path, \
                                                                                                     global_bz, micro_bz, s, n, \
                                                                                                    comm_cost_model_folder=comm_cost_model_dir)
num_hidden_layers = len(T) // 2
num_devices = len(D)

if use_profiler_prediction:
    lat_cost_model.update_profiled_result('/workspace/qpipe/scripts/lat_profiled_result')

available_bits = [2, 3, 4, 8, 16] # cutlass causes illegal memory error for 8:tc and 8:tc-li

for bit in available_bits:
    res_bit = generate_uniform_partition(bit)
    file_name_bit = f'uniform_partition_bit_{bit}_' + model_size + '_' + device_info + '.pkl'
    folder = '/workspace/qpipe/scripts/strategy'
    save_with_pickle(res_bit, file_name_bit, folder)
