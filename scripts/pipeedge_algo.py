
# qllm libs
from qllm.models.OPT.opt import model_cards
# qpipe libs
import qpipe
from qpipe.partitioner.utils import (
    assign_uniform_bit
)
from qpipe.cost_model import (
    estimate_single_layer_mem,
    estimate_all_layer_mem
)
from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    get_slo
)

from qpipe.utils import (
    save_with_pickle
)

# default libs
import pickle
import os 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, required=True)
parser.add_argument('--device_names',  nargs='+', type=str, required=True)
parser.add_argument('--device_numbers',  nargs='+', type=int, required=True)
# adaptive
parser.add_argument('--adaptive', action='store_true')
args = parser.parse_args()

unit = qpipe._globals.MEM_UNIT
time_mult_times = qpipe._globals.TIME_MULT_TIMES

# model size
model_size = args.model_size # '66b'
device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
device_numbers = args.device_numbers # [2, 3]
assert len(device_names) == len(device_numbers), "device_names and device_numbers should have the same length"

import itertools
def transition_equation(h, i, S, j, u, T, D):
    # previous min max throuhput
    minmax_throughput = h[i, frozenset(S), u] # here u is the next rank
    # comp cost
    device_name = D[u]
    t_comp = 0
    for layer_on_u in range(i, j):
        shard = T[layer_on_u]
        bit = bit_assignment[layer_on_u]

        if not use_profiler_prediction:
            t_comp += lat_cost_model.predict_with_hyper(device_name, shard, bit).item()
        else:
            lat = lat_cost_model.predict_with_profiled(device_name, shard, bit)
            if lat is None:
                print(device_name, shard, bit)
                lat = 9999
            t_comp += lat

    # comm cost
    # get last device in S
    t_comm = 0
    device_rank_nums = len(D)
    next_rank = (u+1) % device_rank_nums
    t_comm = comm_cost_model.predict_comm_time(u, next_rank, comm_size)
    res = max(minmax_throughput, t_comp, t_comm)
    return res 

def pipeedge_partition(T, D):
    # D here is rank with keys and values
    D_ranks = list(D.keys())
    L = len(T)
    
    h = {(i, frozenset(S), u): float("inf") for i in range(L + 1) for S in itertools.chain.from_iterable(itertools.combinations(D_ranks, r) for r in range(len(D_ranks) + 1)) for u in D_ranks}
    for u in D_ranks:
        h[(0, frozenset(), u)] = 0 # when previous selection is none, the cost is 0.
    answer = float("inf")
    p = {} # record
    calculation_times = 0
    for i in range(L):
        for S in itertools.chain.from_iterable(itertools.combinations(D_ranks, r) for r in range(len(D_ranks) + 1)):
            for u in set(D_ranks) - set(S): # u is the next device to be used. 
                calculation_times += 1
                print(f"S: {S}, u: {u}, Calculate times: {calculation_times}, answer: {answer}" )
                for j in range(i + 1, L + 1):
                    # check memory constraints
                    # i to j-1 layer. e.g. i=0, j=1, then only layer 0
                    i_to_j_mem = sum(estimate_single_layer_mem(model_mem_estimator, T[k], bit_assignment[k]) for k in range(i, j))
                    device_mem = get_single_device_mem_constraints(D[u])
                    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size(unit='MB')[0] / chunk_size
                    device_mem -= temp_tensor_mem
                    if u == 0:
                        device_mem -= post_pre_mem # first layer need to load data and embedding

                    if i_to_j_mem > device_mem:
                        # print(f"i_to_j_mem: {i_to_j_mem}, device_mem: {device_mem}")
                        break
                    # for device u with j-i layers, new cost: (place j-i layers on u)
                    C = transition_equation(h, i, S, j, u, T, D)
                    if C == float("inf"):
                        break # no need to continue
                    if j == L:
                        if C < answer: # better
                            answer = C
                            index = (i, frozenset(S), u) # precursor of the best solution
                    else:
                        # update v device given the new S.
                        # record the min value of j placement
                        for v in set(D_ranks) - set(S) - {u}:
                            tmp_S = frozenset(set(S).union({u}))
                            print("update result", j-1, tmp_S, v, C, end="\r", flush=True)
                            # the real number will be j - 1
                            if C < h[j-1, tmp_S, v]:
                                h[j-1, tmp_S, v] = C
                                p[j-1, tmp_S, v] = (i, u) # precursor
    test_h = h
    test_p = p
    assert index is not None, "No solution found"
    with open(f'./baseline_result/{file_name}', 'wb') as f: pickle.dump((test_h, test_p, index), f)

def interpret_result(T):
    with open(f'./baseline_result/{file_name}', 'rb') as f: test_h, test_p, index = pickle.load(f)
    L = len(T)
    final_result = {}
    j_idx = index[0]
    previous_device_set = index[1]
    last_device = index[-1]
    final_result[last_device] = [j_idx, L]
    precursor = test_p[index]
    while precursor is not None:
        i, device_rank = precursor
        final_result[device_rank] = [i, j_idx]
        # print(precursor)
        previous_device_set = previous_device_set - set({device_rank})
        new_index = (i, previous_device_set, device_rank)
        if new_index[0] == 0:
            break
        precursor = test_p[new_index]
        j_idx = i
    # sort by rank number
    final_result = {k: final_result[k] for k in sorted(final_result)}
    return final_result

'''
    Initiailization
'''
from qpipe.partitioner import gen_config
# generation configs
global_bz = gen_config.global_bz
micro_bz = gen_config.micro_bz
chunk_size = global_bz // micro_bz
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

available_bits = [2, 3, 4, 8, 16] # pipe edge cannot

adaptive = args.adaptive
if adaptive:
    file_name = f'adaptive_bit_' + model_size + '_' + device_info + '.pkl'
    folder = '/workspace/qpipe/scripts/strategy'
    adapt_file = os.path.join(folder, file_name)
    with open(adapt_file, 'rb') as f:
        result = pickle.load(f)
        bit_assignment = result['bit_assignment']
else:
    bit_assignment = {}
    assign_uniform_bit(T, 8, bit_assignment)

mem_required = estimate_all_layer_mem(model_mem_estimator, T, bit_assignment)
assert mem_required < max_device_mem, "The model is too large to fit in the device mesh"
print(f"Total memory required: {mem_required / 1024 } GB", "available memory: ", max_device_mem / 1024, "GB")
# user input here, press any to continue
input("Press any key to continue")
file_name = 'pipedge_result.pkl' if not adaptive else 'pipedge_result_adaptive.pkl'
result_file_name = 'pipedge_result.pkl' if not adaptive else 'pipedge_result_adaptive.pkl'

post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
pipeedge_partition(T, D)
result = interpret_result(T)
result = {'bit_assignment': bit_assignment, 'partition_result': result}
# save the result
device_info = get_device_info(device_names, device_numbers)
if adaptive:
    file_name = f'pipeedge_adaptive_' + model_size + '_' + device_info + '.pkl'
else:
    file_name = f'pipeedge_' + model_size + '_' + device_info + '.pkl'
folder = '/workspace/qpipe/scripts/strategy'
save_with_pickle(result, file_name, folder)