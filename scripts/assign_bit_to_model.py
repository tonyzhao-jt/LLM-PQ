from qpipe.partitioner import (
    assign_uniform_indicator
)

from qpipe.partitioner.utils import (
    assign_uniform_bit, 
    estimate_min_max_mem,
    create_device_mesh_grid
)

from qpipe.partitioner.assigner import assign_bit_with_mem_constraints

from qpipe.cost_model import (
    estimate_all_layer_mem
)

from qllm.models.OPT.opt import model_cards
from qpipe.partitioner.helper import init_parameters_and_cost_models, get_device_mesh_overall_mem_constraints

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--extra_mem_reduced', type=int, default=0)
args = parser.parse_args()

# device configuration
device_names = ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
# device mesh
device_mesh = {
    0: [4, device_names[1]], # start rank, numbers, device_type
    4: [4, device_names[0]],
}

D = create_device_mesh_grid(device_mesh)
max_device_mem = get_device_mesh_overall_mem_constraints(D)

# read basic configuration from the model_cfg
model_size = '175b'
config = model_cards[model_size]
model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names)

# assign bits
bit_map = {}
assign_uniform_bit(T, 16, bit_map)
initial_mem = estimate_all_layer_mem(model_mem_estimator, T, bit_map)
max_model_mem, min_model_mem = estimate_min_max_mem(model_mem_estimator, T)

if min_model_mem > max_device_mem:
    print(f"Minimum model size {min_model_mem} is larger than device memory {max_device_mem}")
    print('The model is too large to fit in the device memory')
else:
    if initial_mem < max_device_mem:
        print(f"Initial model size {initial_mem} is smaller than device memory {max_device_mem}")
        print('The model is small enough to fit in the device memory')
    else:
        print("start q")
        print(initial_mem, max_device_mem)
        # start quantization
        # assign indicator
        # ILP to get the optimal bit map
        # verysimple, choose bits, minimize the sum of indicator, satsify the constraints of the memory
        # Given L layers: 0,..L-1
        # each layer can have bits b from a set of bits B: 2,4,8,16
        # each layer has a quantization sensitivity indicator i_(l,b) for different quantization bits b
        # the total memory requirement of the model is the sum of the memory requirement of each layer M(l,b)
        # try to minimize the sum of indicator i_(l,b) while satisfying the memory constraint
        available_bits = [2,4,8,16]
        mem_constraints = max_device_mem - args.extra_mem_reduced
        L = len(T)
        indicator = assign_uniform_indicator(L, available_bits)
        assign_bit_with_mem_constraints(T, available_bits, indicator, mem_constraints, model_mem_estimator, verbose=True, store=True,\
                                        file_path='./baseline_result/bit_adaptive.pkl')
