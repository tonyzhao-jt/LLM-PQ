import json
import pickle
from qpipe.partitioner.utils import (
    assign_uniform_bit, 
)



# COMMON PART
from qllm.models.OPT.opt import model_cards
from qpipe.partitioner.utils import (
    create_device_mesh_grid
)

from qpipe.partitioner.helper import init_parameters_and_cost_models

import qpipe
import pickle
from qpipe.cost_model import (
    estimate_single_layer_mem
)
from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    get_device_mesh_overall_mem_constraints
)

unit = qpipe._globals.MEM_UNIT
RATIO_AVOID_OOM = qpipe._globals.RATIO_AVOID_OOM
CUDA_CONTEXT_MEM = qpipe._globals.CUDA_CONTEXT_MEM

# device configuration
device_names = ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
# device mesh
device_mesh = {
    0: [4, device_names[1]], # start rank, numbers, device_type
    4: [4, device_names[0]],
}

D = create_device_mesh_grid(device_mesh)
max_device_mem = get_device_mesh_overall_mem_constraints(D)

# target model configuration
model_size = '175b'
config = model_cards[model_size]
model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default='175b')
# adaptive 
parser.add_argument('--adaptive', action='store_true')
args = parser.parse_args()
adaptive = args.adaptive

# pipeedge result
pipeline_partition_result = "pipedge_result_adaptive.json"
bit_assignment_result = "bit_adaptive.pkl"
if adaptive:
    with open('./baseline_result/bit_adaptive.pkl', 'rb') as f:
        bit_assignment = pickle.load(f)
else:
    bit_assignment = {}
    assign_uniform_bit(T, 8, bit_assignment)

pipeline_partition_result_pipedge = json.load(open(f'./baseline_result/{pipeline_partition_result}'))
bit_assignment_result_pipedge = bit_assignment
# reset the pipeedge_partition_result's key from str to int
pipeline_partition_result_pipedge = {int(k): v for k, v in pipeline_partition_result_pipedge.items()}

# qpipe result
qpipe_result = "qpipe_ilp_result.pkl"
qpipe_result = pickle.load(open(f'./baseline_result/{qpipe_result}', "rb"))
# handle result into pipeegde form
pipeline_partition_result_qpipe = {}
bit_assignment_result_qpipe = {}
L = len(list(qpipe_result.keys()))
for layer, (device_rank, bit) in qpipe_result.items():
    if device_rank not in pipeline_partition_result_qpipe:
        pipeline_partition_result_qpipe[device_rank] = []
        bit_assignment_result_qpipe[device_rank] = []
    pipeline_partition_result_qpipe[device_rank].append(layer)
    bit_assignment_result_qpipe[device_rank].append(bit)

# reset the layer index
for device_rank, layers in pipeline_partition_result_qpipe.items():
    pipeline_partition_result_qpipe[device_rank] = len(layers)
start_idx = 0
for device_rank, layers in pipeline_partition_result_qpipe.items():
    pipeline_partition_result_qpipe[device_rank] = [start_idx, start_idx + layers * 2]
    start_idx += layers * 2

pipeline_partition_result_qpipe = {k: pipeline_partition_result_qpipe[k] for k in sorted(pipeline_partition_result_qpipe)}

available_bits = [2, 4, 8, '8:tc', '8:tc-li', 16] 
available_bits = list(set(available_bits))
BITs = [
    (i, j) for i in available_bits for j in available_bits
]
# assign bits
new_bit_assignment_result_qpipe = {}
for device_rank, bit in bit_assignment_result_qpipe.items():
    part_result = pipeline_partition_result_qpipe[device_rank]
    bit_idx = 0
    for i in range(part_result[0], part_result[1], 2):
        attn_bit, ffn_bit = BITs[bit[bit_idx]]
        new_bit_assignment_result_qpipe[i] = attn_bit
        new_bit_assignment_result_qpipe[i+1] = ffn_bit
        bit_idx += 1
bit_assignment_result_qpipe = new_bit_assignment_result_qpipe

# first make sure the partition is within the memory budget
def check_memory_budget_single_device(device_rank, p_partition_result, p_bit_assign):
    device_name = D[device_rank]
    i, j = p_partition_result[device_rank]
    i_to_j_mem = sum(estimate_single_layer_mem(model_mem_estimator, T[k], p_bit_assign[k]) for k in range(i, j))
    device_mem = get_single_device_mem_constraints(device_name)
    assert i_to_j_mem < device_mem, f"memory budget exceeded for device {device_rank}, {i_to_j_mem} > {device_mem}"

def check_memory_budget(p_partition_result, p_bit_assign, name='qpipe'):
    print("verify memory budget for", name)
    for device_rank in p_partition_result:
        print("device rank passed: ", device_rank)
        check_memory_budget_single_device(device_rank, p_partition_result, p_bit_assign)

check_memory_budget(pipeline_partition_result_qpipe, bit_assignment_result_qpipe)
check_memory_budget(pipeline_partition_result_pipedge, bit_assignment_result_pipedge, name='pipedge')

# then check the latency
# pipedge
# qpipe