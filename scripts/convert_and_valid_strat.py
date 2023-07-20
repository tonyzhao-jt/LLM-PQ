from qllm.models.OPT.opt import model_cards

import shaq 

from shaq.cost_model import (
    estimate_single_layer_mem,
)

from shaq.utils import (
    save_with_pickle
)

from shaq.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    calculate_max_throughputs_and_lat,
    get_device_info
)

# default libs
import pickle
import os 

unit = shaq._globals.MEM_UNIT
time_mult_times = shaq._globals.TIME_MULT_TIMES
slo_rate = shaq._globals.SLO_RATE

# first make sure the partition is within the memory budget
def check_memory_budget_single_device(device_rank, p_partition_result, p_bit_assign):
    device_name = D[device_rank]
    i, j = p_partition_result[device_rank]
    i_to_j_mem = sum(estimate_single_layer_mem(model_mem_estimator, T[k], p_bit_assign[k]) for k in range(i, j))
    device_mem = get_single_device_mem_constraints(device_name)
    tmp_mem = model_mem_estimator.calculate_temp_tensor_size(unit='MB')[0] / chunk_size
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


def log_result(result, name):
    print(f"{name} result: Minimax Lat {result[0]} Throughput (1/Lat): {result[1]} ")


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

partition_strategy_folder = '/workspace/qpipe/scripts/part_strategy'
for name, val in partition_bit_result_dict.items():
    # first check is valid
    if val is None:
        print(f"{name} is not mem valid")
    try:
        partition_result, bit_assignment = val['partition_result'], val['bit_assignment']
        check_memory_budget(partition_result, bit_assignment, name=name)
    except:
        print(f"{name} is not mem valid")
        continue

    partition_result, bit_assignment = val['partition_result'], val['bit_assignment']
    # reinterpret result and calculte the latency
    partition_result, bit_assignment = reset_result_rank_index(partition_result, bit_assignment)
    result = calculate_max_throughputs_and_lat(D, partition_result, bit_assignment, \
                                                 lat_cost_model, comm_cost_model, use_profiler_prediction, comm_size)
    log_result(result, name)
    final_partition_strategies = convert_to_qpipe_result2partitions(partition_result, bit_assignment)
    final_result_file_name = f"{name}_{model_size}_{device_info}_final_strategy.pkl"
    save_with_pickle(final_partition_strategies, final_result_file_name, partition_strategy_folder)

