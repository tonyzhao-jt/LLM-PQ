from qllm.models.OPT.opt import model_cards

import qpipe 
from qpipe.partitioner.indicator import (
    assign_omega_uniform,
)
from qpipe.partitioner.utils import (
    assign_uniform_bit, 
    estimate_min_max_mem,
    interpret_ilp_result_i_j_b
)

from qpipe.cost_model import (
    estimate_single_layer_mem,
    estimate_all_layer_mem
)

from qpipe.utils import (
    save_with_pickle
)

from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info
)

# default libs
import pickle
import os 
import numpy as np 

# setup ilp configs
import pulp
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam('WLSACCESSID',"1b28dca7-337e-4811-b346-01087e09cd64")
env.setParam('WLSSECRET', "629520bd-a114-45d7-b828-bfc5235c198d")
env.setParam('LICENSEID', 965996)
env.start()

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

def solve_ilp_pulp(L, N, BITs, M, M_d, omega):
    prob = pulp.LpProblem("max Latency Minimization Problem", pulp.LpMinimize)
    # Create a new PuLP model
    B = len(BITs)
    z = pulp.LpVariable.dicts("z", [(i, j, b) for i in range(L) for j in range(N) for b in range(B)], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i, b) for i in range(L) for b in range(B)], cat=pulp.LpBinary)

    # Define the objective function
    prob += pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)])

    # Define the constraints
    for i in range(L):
        prob += pulp.lpSum([z[(i, j, b)] for j in range(N) for b in range(B)]) == 1
    for i in range(L):
        for b in range(B):
            prob += pulp.lpSum([z[(i, j, b)] for j in range(N)]) == y[(i, b)]
    for j in range(N):
        prob += pulp.lpSum([z[(i, j, b)] * M[i][b] for i in range(L) for b in range(B)]) <= M_d[j]
    
    # Solve the problem
    prob.solve(pulp.GUROBI(MIPGap=0.004))
    # prob.solve(pulp.GUROBI())
    # prob.solve()
    # Print the solution status
    print("Status:", pulp.LpStatus[prob.status])
    # Print the optimal objective value
    print("Optimal value of the objective function:", pulp.value(prob.objective))
    # store the optimal solution
    result = {}
    # print z variable result
    for i in range(L):
        for j in range(N):
            for b in range(B):
                if z[(i, j, b)].varValue > 0:
                    print("z[{}, {}, {}] = {}".format(i, j, b, z[(i, j, b)].varValue))
                    result[i] = (j, b)
    return result


def get_mem_with_layer_bit_pair(bit_pairs): 
    mem_bits_vector = np.zeros(len(bit_pairs))
    for idx, bit_pair in enumerate(bit_pairs):
        attn_bit, ffn_bit = bit_pair
        attn_mem = estimate_single_layer_mem(model_mem_estimator, 0, attn_bit)
        ffn_mem = estimate_single_layer_mem(model_mem_estimator, 1, ffn_bit)
        mem = attn_mem + ffn_mem
        mem_bits_vector[idx] = mem
    return mem_bits_vector

def prepare_for_ilp(num_hidden_layers, D, available_bits):
    L = num_hidden_layers # in partition, regard as a whole
    N = len(D) # number of devices
    available_bits = list(set(available_bits))
    BITs = [
        (i, j) for i in available_bits for j in available_bits
    ]
    M_d = np.array([get_single_device_mem_constraints(device_name) for d_rank, device_name in D.items()]) 
    mem_bits_vector = get_mem_with_layer_bit_pair(BITs)
    M = np.tile(mem_bits_vector, (L, 1))
    # reduce the embedding size on device 0
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size(unit='MB')[0] / chunk_size
    M_d[0] -= post_pre_mem
    M_d -= time_mult_times * temp_tensor_mem # torch may not release the tensor immediately, modify the 2 to ensure won't oom
    M_d = np.floor(M_d).astype(int) # floor
    M = np.ceil(M).astype(int) # ceil
    # omega
    # omega = assign_omega_constant(L, BITs)
    omega = assign_omega_uniform(L, BITs)
    return L, N, BITs, M_d, M, omega


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

# assign bits
bit_map = {}
assign_uniform_bit(T, 16, bit_map)
initial_mem = estimate_all_layer_mem(model_mem_estimator, T, bit_map)
max_model_mem, min_model_mem = estimate_min_max_mem(model_mem_estimator, T)

num_hidden_layers = len(T) // 2
num_devices = len(D)

# set available bits
available_bits = [2, 4, 8, 16]

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
        L, N, BITs, M_d, M, omega = prepare_for_ilp(num_hidden_layers, D, available_bits)
        result = solve_ilp_pulp(L, N, BITs, M, M_d, omega)
        result = interpret_ilp_result_i_j_b(result, available_bits)

        device_info = get_device_info(device_names, device_numbers)

        file_name = f'adaptive_bit_' + model_size + '_' + device_info + '.pkl'
        folder = '/workspace/qpipe/scripts/strategy'
        save_with_pickle(result, file_name, folder)
