from qllm.models.OPT.opt import model_cards

import qpipe 

from qpipe.cost_model import (
    estimate_single_layer_mem,
)

from qpipe.utils import (
    save_with_pickle
)

from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    calculate_max_throughputs_and_lat,
    get_device_info
)

# default libs
import pickle


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, required=True)
parser.add_argument('--device_names',  nargs='+', type=str, required=True)
parser.add_argument('--device_numbers',  nargs='+', type=int, required=True)
parser.add_argument('--SLO-aware',  action='store_true', help='add slo into constraints')
parser.add_argument('--adaptive', action='store_true')
args = parser.parse_args()

unit = qpipe._globals.MEM_UNIT
time_mult_times = qpipe._globals.TIME_MULT_TIMES
slo_rate = qpipe._globals.SLO_RATE

# model size
model_size = args.model_size # '66b'
device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
device_numbers = args.device_numbers # [2, 3]
slo_aware = args.SLO_aware
adaptive = args.adaptive
assert len(device_names) == len(device_numbers), "device_names and device_numbers should have the same length"

'''
    Initiailization
'''
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

# strategy folder
folder = '/workspace/qpipe/scripts/strategy'
device_info = get_device_info(device_names, device_numbers)

partition_bit_result_dict = {}
target_test_strategies = ['qpipe', 'qpipe_slo', 'pipeedge', 'pipeedge_adaptive', 'adabit', 'uniform']
# target_test_strategies = ['qpipe', 'adabit', 'uniform']

'''
    PipeEdge Result
'''
if 'pipeedge' in target_test_strategies:
    if adaptive:
        file_name = f'pipeedge_adaptive_' + model_size + '_' + device_info + '.pkl'
        pipe_edge_strategy = pickle.load(open(f'{folder}/{file_name}', 'rb'))
        partition_bit_result_dict['pipeedge_adaptive'] = pipe_edge_strategy
    else:
        file_name = f'pipeedge_' + model_size + '_' + device_info + '.pkl'
        pipe_edge_strategy = pickle.load(open(f'{folder}/{file_name}', 'rb'))
        partition_bit_result_dict['pipeedge'] = pipe_edge_strategy

'''
    QPipe Result
'''
if 'qpipe' in target_test_strategies:
    if slo_aware:
        file_name = f'adaqpipe_slo_' + model_size + '_' + device_info + '.pkl'
        qpipe_result = pickle.load(open(f'{folder}/{file_name}', 'rb'))
        partition_bit_result_dict['qpipe_slo'] = qpipe_result
    else:
        file_name = f'adaqpipe_' + model_size + '_' + device_info + '.pkl'
        qpipe_result = pickle.load(open(f'{folder}/{file_name}', 'rb'))
        partition_bit_result_dict['qpipe'] = qpipe_result
    
'''
    QPipe Sole.
'''
'''
    BitAdaptive Sole.
'''
if 'adabit' in target_test_strategies:
    # adaptive bit result
    file_name = f'adaptive_bit_' + model_size + '_' + device_info + '.pkl'
    adaptive_result = pickle.load(open(f'{folder}/{file_name}', 'rb'))
    partition_bit_result_dict['adabit'] = adaptive_result

'''
    Uniform results
'''
if 'uniform' in target_test_strategies:
    for bit in [2, 3, 4, 8, 16]:
        file_name_bit = f'uniform_partition_bit_{bit}_' + model_size + '_' + device_info + '.pkl'
        uniform_bit_result = pickle.load(open(f'{folder}/{file_name_bit}', 'rb'))
        partition_bit_result_dict[f'uniform_{bit}'] = uniform_bit_result

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
    final_result_file_name = f"{name}_{device_info}_final_strategy.pkl"
    save_with_pickle(final_partition_strategies, final_result_file_name, partition_strategy_folder)

