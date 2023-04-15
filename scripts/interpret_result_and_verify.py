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

use_profiler_prediction = True
# target model configuration
model_size = '175b'
config = model_cards[model_size]
model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names)

if use_profiler_prediction:
    lat_cost_model.update_profiled_result('/workspace/qpipe/scripts')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default='175b')
# adaptive 
parser.add_argument('--adaptive', action='store_true')
args = parser.parse_args()
adaptive = args.adaptive

# pipeedge result (int8)
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
pipeline_partition_result_qpipe, bit_assignment_result_qpipe = qpipe_result['partition_result'], qpipe_result['bit_assignment']

'''
    Just look some other results also.
'''
# adaptive bit result
adaptive_result = "bit_adaptive.pkl"
adaptive_result = pickle.load(open(f'./baseline_result/{adaptive_result}', "rb"))
pipeline_partition_result_adabit, bit_assignment_result_adabit = adaptive_result['partition_result'], adaptive_result['bit_assignment']

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
        # print("device rank passed: ", device_rank)
        check_memory_budget_single_device(device_rank, p_partition_result, p_bit_assign)
    print("all passed")

check_memory_budget(pipeline_partition_result_qpipe, bit_assignment_result_qpipe)
check_memory_budget(pipeline_partition_result_pipedge, bit_assignment_result_pipedge, name='pipedge')
check_memory_budget(pipeline_partition_result_adabit, bit_assignment_result_adabit, name='adabit')

# then evalute the end-to-end latency and throughputs for different methods
# pipedge
# qpipe

def calculate_max_throughputs_and_lat(D, p_partition_result, p_bit_assign):
    e2e_lat = 0
    minmax_throughputs = 0
    max_rank = len(D)
    for d_rank, device_name in D.items():
        layers_start, layers_end = p_partition_result[d_rank]
        comp_time = 0
        # latency
        for layer in range(layers_start, layers_end):
            shard = layer % 2
            bit = p_bit_assign[layer]
            if not use_profiler_prediction:
                comp_time += lat_cost_model.predict_with_hyper(device_name, shard, bit).item()
            else:
                lat = lat_cost_model.predict_with_profiled(device_name, shard, bit)
                comp_time += lat
        # communication
        next_rank = (d_rank + 1) % max_rank
        t_comm = comm_cost_model.predict_comm_time(d_rank, next_rank, comm_size)
        # minmax throughput
        minmax_throughputs = max(minmax_throughputs, comp_time, t_comm)
        e2e_lat += max(comp_time, t_comm)
    return minmax_throughputs, e2e_lat

def log_result(result, name):
    print(f"{name} result: Throughputs {1/result[0]} Lat {result[1]} ")

qpipe_result = calculate_max_throughputs_and_lat(D, pipeline_partition_result_qpipe, bit_assignment_result_qpipe)
pipedge_result = calculate_max_throughputs_and_lat(D, pipeline_partition_result_pipedge, bit_assignment_result_pipedge)
adabit_result = calculate_max_throughputs_and_lat(D, pipeline_partition_result_adabit, bit_assignment_result_adabit)

log_result(qpipe_result, 'qpipe')
log_result(pipedge_result, 'pipedge')
log_result(adabit_result, 'adabit')