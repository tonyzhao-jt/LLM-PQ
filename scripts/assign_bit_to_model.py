from qpipe.partitioner import (
    assign_uniform_indicator
)

from qpipe.partitioner.utils import (
    assign_uniform_bit, 
    estimate_min_max_mem,
    get_maximum_available_mem,
    create_device_mesh_grid
)

from qpipe.partitioner.assigner import assign_bit_with_mem_constraints

from qpipe.cost_model import (
    estimate_all_layer_mem
)
import numpy as np 
import qpipe

from qllm.models.OPT.opt import model_cards
from qllm.utils import ModelMemEstimator


device_mesh = {
    0: [4, 'NVIDIA_A100-SXM4-40GB'], # start rank, numbers, device_type
    4: [4, 'Tesla_V100-SXM2-32GB'],
}

unit = qpipe._globals.MEM_UNIT
RATIO_TO_AVOID_OOM = qpipe._globals.RATIO_TO_AVOID_OOM
CUDA_CONTEXT_MEM = qpipe._globals.CUDA_CONTEXT_MEM
D = create_device_mesh_grid(device_mesh)
max_device_mem = get_maximum_available_mem(device_mesh)

# read basic configuration from the model_cfg
model_size = '175b'
config = model_cards[model_size]
h1 = config.hidden_size
h2 = config.ffn_dim
num_hidden_layers = config.num_hidden_layers # decoder layer numbers

# set the chunk size
b = 16
# set the prompt sequence length
s = 512
# set the number of generated tokens
n = 100

model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n)
hidden_space_numels = model_mem_estimator.estimate_hidden_space()

# T equals to num_hidden_layers, 0,1
T = [0,1] * num_hidden_layers
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
        mem_constraints = RATIO_TO_AVOID_OOM * max_device_mem
        L = len(T)
        indicator = assign_uniform_indicator(L, available_bits)
        assign_bit_with_mem_constraints(T, available_bits, indicator, mem_constraints, model_mem_estimator, verbose=True, store=True,\
                                        file_path='./baseline_result/bit_adaptive.pkl')
