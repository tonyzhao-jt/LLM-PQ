from qpipe.partitioner.indicator import (
    assign_omega_uniform,
    assign_omega_constant
)
from qpipe.partitioner.utils import (
    assign_uniform_bit, 
    estimate_min_max_mem,
    create_device_mesh_grid,
    interpret_ilp_result_i_j_b
)

from qpipe.cost_model import (
    estimate_single_layer_mem,
    estimate_all_layer_mem
)

from qllm.models.OPT.opt import model_cards
import numpy as np 
from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    get_device_mesh_overall_mem_constraints,
    
)

import argparse
import qpipe
parser = argparse.ArgumentParser()
# parser.add_argument('--extra_mem_reduced', type=int, default=0)
args = parser.parse_args()

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

# read basic configuration from the model_cfg
model_size = '175b'
config = model_cards[model_size]
model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names)

# assign bits
bit_map = {}
assign_uniform_bit(T, 16, bit_map)
initial_mem = estimate_all_layer_mem(model_mem_estimator, T, bit_map)
max_model_mem, min_model_mem = estimate_min_max_mem(model_mem_estimator, T)

hidden_layer_num = len(T) // 2

import pulp
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam('WLSACCESSID',"1b28dca7-337e-4811-b346-01087e09cd64")
env.setParam('WLSSECRET', "629520bd-a114-45d7-b828-bfc5235c198d")
env.setParam('LICENSEID', 965996)
env.start()

avaliable_solvers = pulp.list_solvers(onlyAvailable=True)
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
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size(unit='MB')[0]
    M_d[0] -= post_pre_mem
    M_d -= time_mult_times * temp_tensor_mem
    M_d = M_d.astype(int)
    M = M.astype(int)
    # omega
    # omega = assign_omega_constant(L, BITs)
    omega = assign_omega_uniform(L, BITs)
    return L, N, BITs, M_d, M, omega

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
        available_bits = [2, 4, 8, 16]
        L, N, BITs, M_d, M, omega = prepare_for_ilp(hidden_layer_num, D, available_bits)
        result = solve_ilp_pulp(L, N, BITs, M, M_d, omega)
        result = interpret_ilp_result_i_j_b(result, available_bits)
        # store result
        result_file_name = 'bit_adaptive.pkl'
        root_path = './baseline_result'
        import os, pickle 
        result_path = os.path.join(root_path, result_file_name)
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
