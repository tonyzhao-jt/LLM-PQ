
from qllm.models.OPT.opt import model_cards
from qllm.utils import ModelMemEstimator
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
    estimate_all_layer_mem, estimate_single_layer_mem
)

from qpipe.utils import get_device_mem_offline

from qpipe.cost_model import CommCostModel, LatCostModel
from qpipe.utils import get_size_cpu
import qpipe
import torch
import pickle

unit = qpipe._globals.MEM_UNIT
RATIO_TO_AVOID_OOM = qpipe._globals.RATIO_TO_AVOID_OOM
CUDA_CONTEXT_MEM = qpipe._globals.CUDA_CONTEXT_MEM

# device configuration
device_names = ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
# device mesh
device_mesh = {
    0: [4, device_names[1]], # start rank, numbers, device_type
    4: [4, device_names[0]],
}

D = create_device_mesh_grid(device_mesh)

# get the maximum device memory
max_device_mem = get_maximum_available_mem(device_mesh)

# target model configuration
model_size = '175b'
config = model_cards[model_size]
h1 = config.hidden_size
h2 = config.ffn_dim
num_hidden_layers = config.num_hidden_layers # decoder layer numbers

# micro_batch_size
b = 16
# set the prompt sequence length
s = 512
# set the number of generated tokens
n = 100

# T equals to num_hidden_layers, 0,1
T = [0,1] * num_hidden_layers

# estimators
model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n)
comm_size = (b * 1 * h1 * 2) / 1024 / 1024 # MB

# cost models
cost_model_store_path = '/workspace/qpipe/scripts/lat_cost_model'
comm_cost_model = CommCostModel(comm_cost_model_folder='/workspace/qpipe/scripts/comm_cost_model/')
lat_cost_model = LatCostModel(cost_model_store_path, device_names)
lat_cost_model.register_hyper_params(b, s+n, h1, h2)
# comm_cost_model.print_model_available_keys()
# comm_cost = comm_cost_model.predict_comm_time(start_rank=0, end_rank=1, data_size=get_size_cpu(x, unit='MB'))
# predicted_cost = lat_cost_model.predict(device, shard, b, i, h1, h2, bit)
# load bit assignment
adaptive = False
if adaptive:
    with open('bit_adaptive.pkl', 'rb') as f:
        bit_assignment = pickle.load(f)
else:
    bit_assignment = {}
    assign_uniform_bit(T, 8, bit_assignment)
    mem_required = estimate_all_layer_mem(model_mem_estimator, T, bit_assignment)
    assert mem_required < (RATIO_TO_AVOID_OOM * max_device_mem - len(D) * CUDA_CONTEXT_MEM), "The model is too large to fit in the device mesh"

file_name = 'pipedge_result.pkl' if not adaptive else 'pipedge_result_adaptive.pkl'
result_file_name = 'pipedge_result.txt' if not adaptive else 'pipedge_result_adaptive.txt'

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
        t_comp += lat_cost_model.predict_with_hyper(device_name, shard, bit).item()
    # comm cost
    # get last device in S
    t_comm = 0
    if len(S) != 0:
        last_device = list(S)[-1]
        t_comm = comm_cost_model.predict_comm_time(last_device, u, comm_size)
    
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
                print(f"S: {S}, u: {u}, Calculate times: {calculation_times}, answer: {answer}")
                for j in range(i + 1, L + 1):
                    # check memory constraints
                    # i to j-1 layer. e.g. i=0, j=1, then only layer 0
                    i_to_j_mem = sum(estimate_single_layer_mem(model_mem_estimator, T[k], bit_assignment[k]) for k in range(i, j))
                    device_mem = RATIO_TO_AVOID_OOM * get_device_mem_offline(D[u], unit=unit) - CUDA_CONTEXT_MEM

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
                            print("update result", j-1, tmp_S, v, C)
                            # the real number will be j - 1
                            if C < h[j-1, tmp_S, v]:
                                h[j-1, tmp_S, v] = C
                                p[j-1, tmp_S, v] = (i, u) # precursor
    test_h = h
    test_p = p
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
    print(final_result)
    import json
    with open(f'./baseline_result/{result_file_name}', 'w') as f:
        json.dump(final_result, f)
    # times: 196608, answer: 0.34138697775843824, adaptive
    return final_result
pipeedge_partition(T, D)
interpret_result(T)