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
    get_device_mesh_overall_mem_constraints,
    calculate_max_throughputs_and_lat
)

unit = qpipe._globals.MEM_UNIT
RATIO_AVOID_OOM = qpipe._globals.RATIO_AVOID_OOM
CUDA_CONTEXT_MEM = qpipe._globals.CUDA_CONTEXT_MEM
time_mult_times = qpipe._globals.TIME_MULT_TIMES

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
    lat_cost_model.update_profiled_result('/workspace/qpipe/scripts/lat_profiled_result')

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
    tmp_mem = model_mem_estimator.calculate_temp_tensor_size(unit='MB')[0]
    device_mem -= time_mult_times * tmp_mem
    if device_rank == 0:
        post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
        device_mem = device_mem - post_pre_mem 
    print(i_to_j_mem, device_mem)
    assert i_to_j_mem < device_mem, f"memory budget exceeded for device {device_rank}, {i_to_j_mem} > {device_mem}"

def check_memory_budget(p_partition_result, p_bit_assign, name='qpipe'):
    print("verify memory budget for", name)
    for device_rank in p_partition_result:
        # print("device rank passed: ", device_rank)
        check_memory_budget_single_device(device_rank, p_partition_result, p_bit_assign)
    print("all passed")

check_memory_budget(pipeline_partition_result_qpipe, bit_assignment_result_qpipe)
# check_memory_budget(pipeline_partition_result_pipedge, bit_assignment_result_pipedge, name='pipedge')
check_memory_budget(pipeline_partition_result_adabit, bit_assignment_result_adabit, name='adabit')
# then evalute the end-to-end latency and throughputs for different methods
# pipedge
# qpipe


def log_result(result, name):
    print(f"{name} result: Throughputs {1/result[0]} Lat {result[1]} ")


# convert to the result can be used by qpipe
def convert_to_qpipe_result2partitions(pipeline_partition_result, bit_assignment_result):
    # result is something like
    '''
        sharding_strategy = {
        0: {},
        1: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
        },
        2: {
            8: {'shard': [1], 'bits': [16]},
            9: {'shard': [0,1], 'bits': [16, 16]},
            10: {'shard': [0,1], 'bits': [8, 16]},
        },
    }
    '''
    partition_strategies = {}
    layer_num = 0
    for device_rank, (i, j) in pipeline_partition_result.items():
        # device rank use int
        partition_strategies[device_rank] = {}
        for shard_layer_num in range(i, j):
            if shard_layer_num % 2 == 0:
                # self attn
                # each two layers are in the same partition
                partition_strategies[device_rank][layer_num] = {'shard': [0], 'bits': [bit_assignment_result[shard_layer_num]]}
            elif shard_layer_num % 2 == 1:
                # ffn
                if not layer_num in partition_strategies[device_rank]: 
                    partition_strategies[device_rank][layer_num] = {'shard': [1], 'bits':[bit_assignment_result[shard_layer_num]]}
                else:
                    partition_strategies[device_rank][layer_num]['shard'].append(1)
                    partition_strategies[device_rank][layer_num]['bits'].append(bit_assignment_result[shard_layer_num])
                layer_num += 1 # only when ffn is done, we move to the next layer

    return partition_strategies

# cases: {0: [93, 119], 1: [46, 72], 2: [139, 166], 3: [166, 192], 4: [0, 24], 5: [24, 46], 6: [72, 93], 7: [119, 139]}
def reset_result_rank_index(p_pipeline_partition_result_pipedge, bit_assignment_result_pipedge):
    new_result = {}
    new_bit_assignment_result = {}
    new_idx = 0
    for device_rank, (i, j) in p_pipeline_partition_result_pipedge.items():
        layer_num = j - i
        new_result[device_rank] = [new_idx, new_idx + layer_num]
        for k in range(layer_num):
            new_bit_assignment_result[new_idx + k] = bit_assignment_result_pipedge[i + k]
        new_idx += layer_num
    return new_result, new_bit_assignment_result

pipeline_partition_result_pipedge, bit_assignment_result_pipedge = reset_result_rank_index(pipeline_partition_result_pipedge, bit_assignment_result_pipedge)
pipeline_partition_result_adabit, bit_assignment_result_adabit = reset_result_rank_index(pipeline_partition_result_adabit, bit_assignment_result_adabit)
pipeline_partition_result_qpipe, bit_assignment_result_qpipe = reset_result_rank_index(pipeline_partition_result_qpipe, bit_assignment_result_qpipe)

print("pipeline_partition_result_pipedge", pipeline_partition_result_pipedge)
print("pipeline_partition_result_adabit", pipeline_partition_result_adabit)
print("pipeline_partition_result_qpipe", pipeline_partition_result_qpipe)

# simulator result
qpipe_result = calculate_max_throughputs_and_lat(D, pipeline_partition_result_qpipe, bit_assignment_result_qpipe, \
                                                 lat_cost_model, comm_cost_model, use_profiler_prediction, comm_size)
pipedge_result = calculate_max_throughputs_and_lat(D, pipeline_partition_result_pipedge, bit_assignment_result_pipedge, \
                                                    lat_cost_model, comm_cost_model, use_profiler_prediction, comm_size)
adabit_result = calculate_max_throughputs_and_lat(D, pipeline_partition_result_adabit, bit_assignment_result_adabit, \
                                                    lat_cost_model, comm_cost_model, use_profiler_prediction, comm_size)

log_result(qpipe_result, 'qpipe')
log_result(pipedge_result, 'pipedge')
log_result(adabit_result, 'adabit')


qpipe_partition_strategies = convert_to_qpipe_result2partitions(pipeline_partition_result_qpipe, bit_assignment_result_qpipe)
pipedge_partition_strategies = convert_to_qpipe_result2partitions(pipeline_partition_result_pipedge, bit_assignment_result_pipedge)
adabit_partition_strategies = convert_to_qpipe_result2partitions(pipeline_partition_result_adabit, bit_assignment_result_adabit)

# qpipe partition strategy result
pipeline_strategy_result_qpipe = "pipeline_strategy_result_qpipe.pkl"
pipeline_strategy_result_pipedge = "pipeline_strategy_result_pipedge.pkl"
pipeline_strategy_result_adabit = "pipeline_strategy_result_adabit.pkl"
# store the partition strategies
pickle.dump(qpipe_partition_strategies, open(f'./baseline_result/{pipeline_strategy_result_qpipe}', "wb"))
pickle.dump(pipedge_partition_strategies, open(f'./baseline_result/{pipeline_strategy_result_pipedge}', "wb"))
pickle.dump(adabit_partition_strategies, open(f'./baseline_result/{pipeline_strategy_result_adabit}', "wb"))
