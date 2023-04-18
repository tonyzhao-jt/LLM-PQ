# qpipe algorithm has a hyper parameter theta to determine the concern for the precision
# higher theta means more concern for the precision, lower theta means more concern for the latency/throughput
from qllm.models.OPT.opt import model_cards
from qpipe.partitioner.indicator import (
    assign_omega_uniform
)
from qpipe.partitioner.utils import (
    create_device_mesh_grid,
    interpret_ilp_result_i_j_b
)

from qpipe.cost_model import (
    estimate_single_layer_mem
)

import qpipe
import pickle
import numpy as np 
from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    get_device_mesh_overall_mem_constraints,
    calculate_max_throughputs_and_lat
)

from qpipe.cost_model import price as price_model

unit = qpipe._globals.MEM_UNIT

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
num_hidden_layers = len(T) // 2
# comm_cost_model.print_model_available_keys()
# comm_cost = comm_cost_model.predict_comm_time(start_rank=0, end_rank=1, data_size=get_size_cpu(x, unit='MB'))
# predicted_cost = lat_cost_model.predict(device, shard, b, i, h1, h2, bit)

if use_profiler_prediction:
    lat_cost_model.update_profiled_result('/workspace/qpipe/scripts/lat_profiled_result')

file_name = 'qpipe_result.pkl' 
result_file_name = 'qpipe_result.txt'
available_bits = [2, 4, 8, '8:tc', '8:tc-li', 16] # we now can do hardware-aware quantization with 8:tc and 8:tc-li


def get_mem_available_devices(T, D, allocation_schemes, bit_assignment):
    if len(allocation_schemes) == 0: return D 
    # for each device, we need to estimate the memory usage
    est_mem_usage = {}
    for idx, device_rank in enumerate(allocation_schemes):
        shard = T[idx]
        bit = bit_assignment[idx]
        mem = estimate_single_layer_mem(model_mem_estimator, shard, bit)
        if device_rank not in est_mem_usage:
            est_mem_usage[device_rank] = mem
        else:
            est_mem_usage[device_rank] += mem
    
    # check memory usage for each device
    available_devices = {}
    for rank, device_name in enumerate(D):
        device_mem = get_single_device_mem_constraints(device_name)
        if rank not in est_mem_usage:
            available_devices[rank] = device_mem
        else:
            if est_mem_usage[rank] < device_mem:
                available_devices[rank] = device_mem - est_mem_usage[rank]
    return available_devices

import pulp
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam('WLSACCESSID',"1b28dca7-337e-4811-b346-01087e09cd64")
env.setParam('WLSSECRET', "629520bd-a114-45d7-b828-bfc5235c198d")
env.setParam('LICENSEID', 965996)
env.start()

def solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta):
    prob = pulp.LpProblem("max Latency Minimization Problem", pulp.LpMinimize)
    # Create a new PuLP model
    B = len(BITs)
    z = pulp.LpVariable.dicts("z", [(i, j, b) for i in range(L) for j in range(N) for b in range(B)], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i, b) for i in range(L) for b in range(B)], cat=pulp.LpBinary)
    LAT = pulp.LpVariable.dicts("LAT", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous)
    LAT_max = pulp.LpVariable("LAT_max", lowBound=0, cat=pulp.LpContinuous)

    # Define the objective function
    prob += LAT_max + theta * pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)])

    # Define the constraints
    for i in range(L):
        prob += pulp.lpSum([z[(i, j, b)] for j in range(N) for b in range(B)]) == 1
    for i in range(L):
        for b in range(B):
            prob += pulp.lpSum([z[(i, j, b)] for j in range(N)]) == y[(i, b)]
    for j in range(N):
        prob += pulp.lpSum([z[(i, j, b)] * M[i][b] for i in range(L) for b in range(B)]) <= M_d[j]
        prob += pulp.lpSum([z[(i, j, b)] * l[i][j][b] for i in range(L) for b in range(B)]) <= LAT[j]
        prob += LAT[j] >= comm[j]
        prob += LAT_max >= LAT[j]
    
    # Solve the problem
    # prob.solve(pulp.apis.PULP_CBC_CMD(msg=0))
    prob.solve()

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
                    # print("z[{}, {}, {}] = {}".format(i, j, b, z[(i, j, b)].varValue))
                    result[i] = (j, b)
    return result, pulp.value(prob.objective)


def solve_ilp_pulp_with_price(L, N, BITs, M, M_d, l, omega, comm, price, theta, gamma):
    prob = pulp.LpProblem("max Latency Minimization Problem", pulp.LpMinimize)
    # Create a new PuLP model
    B = len(BITs)
    z = pulp.LpVariable.dicts("z", [(i, j, b) for i in range(L) for j in range(N) for b in range(B)], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i, b) for i in range(L) for b in range(B)], cat=pulp.LpBinary)
    x = pulp.LpVariable.dicts("x", [j for j in range(N)], cat=pulp.LpBinary)
    LAT = pulp.LpVariable.dicts("LAT", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous)
    LAT_max = pulp.LpVariable("LAT_max", lowBound=0, cat=pulp.LpContinuous)

    # Define the objective function
    prob += LAT_max + theta * pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)]) + \
    gamma * pulp.lpSum(x[j] * price[j] for j in range(N))

    # Define the constraints
    for i in range(L):
        prob += pulp.lpSum([z[(i, j, b)] for j in range(N) for b in range(B)]) == 1
    for i in range(L):
        for b in range(B):
            prob += pulp.lpSum([z[(i, j, b)] for j in range(N)]) == y[(i, b)]

    for j in range(N):
        prob += pulp.lpSum([z[(i, j, b)] * M[i][b] for i in range(L) for b in range(B)]) <= M_d[j]
        prob += pulp.lpSum([z[(i, j, b)] * l[i][j][b] for i in range(L) for b in range(B)]) <= LAT[j]
        # for device on 0: master, need to add the embedding time
        prob += LAT[j] >= comm[j]
        prob += LAT_max >= LAT[j]

        # x[j] = 1 if the task j is assigned to a device
        prob += x[j] >= pulp.lpSum([z[(i, j, b)] for i in range(L) for b in range(B)])
        prob += x[j] <= pulp.lpSum([z[(i, j, b)] for i in range(L) for b in range(B)])
        
    # Solve the problem
    prob.solve(pulp.apis.PULP_CBC_CMD(msg=0))

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
                    # print("z[{}, {}, {}] = {}".format(i, j, b, z[(i, j, b)].varValue))
                    result[i] = (j, b)
    # get whether device j is used
    device_used = []
    for j in range(N):
        if x[j].varValue > 0:
            device_used.append(j)
        
    return result, pulp.value(prob.objective), device_used
    
def get_mem_with_layer_bit_pair(bit_pairs): 
    mem_bits_vector = np.zeros(len(bit_pairs))
    for idx, bit_pair in enumerate(bit_pairs):
        attn_bit, ffn_bit = bit_pair
        attn_mem = estimate_single_layer_mem(model_mem_estimator, 0, attn_bit)
        ffn_mem = estimate_single_layer_mem(model_mem_estimator, 1, ffn_bit)
        mem = attn_mem + ffn_mem
        mem_bits_vector[idx] = mem
    return mem_bits_vector

def get_latency_with_layer_device_bit_pair(D, bit_pairs):
    device_names = list(D.values())
    dtypes = set(device_names)
    device_bit_res = {}

    for device_name in dtypes:
        for idx, bit_pair in enumerate(bit_pairs):
            attn_bit, ffn_bit = bit_pair
            device_bit_res[(device_name, bit_pair)] = 0
            if not use_profiler_prediction:
                attn_lat = lat_cost_model.predict_with_hyper(device_name, 0, attn_bit).item()
                ffn_lat = lat_cost_model.predict_with_hyper(device_name, 1, ffn_bit).item()
            else:
                attn_lat = lat_cost_model.predict_with_profiled(device_name, 0, attn_bit)
                ffn_lat = lat_cost_model.predict_with_profiled(device_name, 1, ffn_bit)
                if attn_lat is None: # bit is not available
                    attn_lat = 9999 # a large number
                    # print(device_name, 0, attn_bit)
                if ffn_lat is None:
                    ffn_lat = 9999
                    # print(device_name, 1, ffn_bit)
            lat = attn_lat + ffn_lat
            device_bit_res[(device_name, bit_pair)] = lat
    # create latency matrix
    lat_device_bits_matrix = np.zeros((len(D), len(bit_pairs)))
    for i, device_name in enumerate(device_names):
        for j, bit_pair in enumerate(bit_pairs):
            lat_device_bits_matrix[i, j] = device_bit_res[(device_name, bit_pair)]
    return lat_device_bits_matrix

def get_comm(D):
    device_length = len(D)
    comm = np.zeros(device_length)
    for idx in range(device_length):
        comm[idx] = comm_cost_model.predict_comm_time(idx, (idx + 1) % device_length, comm_size)
    return comm


def prepare_for_ilp(num_hidden_layers, D, available_bits):
    L = num_hidden_layers # in partition, regard as a whole
    N = len(D) # number of devices

    available_bits = list(set(available_bits))
    BITs = [
        (i, j) for i in available_bits for j in available_bits
    ]

    '''
        Constraint related
    '''
    # device memory
    M_d = np.array([get_single_device_mem_constraints(device_name) for d_rank, device_name in D.items()]) 
    # construct bit memory matrix
    # M = np.zeros((L, len(BITs)))
    # bit_pairs = BITs
    # mem_bits_vector = get_mem_with_layer_bit_pair(bit_pairs)
    # for i in range(L):
    #     M[i, :] = mem_bits_vector
    
    mem_bits_vector = get_mem_with_layer_bit_pair(BITs)
    M = np.tile(mem_bits_vector, (L, 1))

    # reduce the embedding size on device 0 for M_d
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
    M_d[0] -= post_pre_mem
    # latency
    l = np.zeros((L, N, len(BITs)))
    lat_device_bits_matrix = get_latency_with_layer_device_bit_pair(D, BITs)
    for i in range(L):
        l[i, :, :] = lat_device_bits_matrix
    
    # omega
    omega = assign_omega_uniform(L, BITs)

    # comm
    comm = get_comm(D)

    # hyperparameters
    theta = 0.1

    # price related
    gamma = 0.01
    # get price
    price = np.zeros(N)
    for i in range(N):
        price[i] = price_model.get_price(D[i])
    return L, N, BITs, M_d, M, l, omega, comm, price, theta, gamma
    
# solve_ilp(L, N, BITs, M, M_d, l, omega, comm, theta)
# add chunk searching
available_chunks = lat_cost_model.get_available_chunks()
# test chunk spliting.
# max_throughput = 0
# for chunk in available_chunks: # chunk with number of chunks and bs
#     num_chunks, bs = chunk
#     lat_cost_model.change_bs(bs) # change lat cost model is enough, maximum memory won't change.
#     L, N, BITs, M_d, M, l, omega, comm, theta = prepare_for_ilp(num_hidden_layers, D, available_bits)
#     result, obj_value = solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta)
#     result = interpret_ilp_result_i_j_b(result, available_bits)
#     partition_result, bit_assignment = result['partition_result'], result['bit_assignment']
#     minmax_throughputs, e2e_lat = calculate_max_throughputs_and_lat(D, partition_result, bit_assignment, \
#                                       lat_cost_model, comm_cost_model, use_profiler_prediction, comm_size)
#     minmax_throughputs = bs * minmax_throughputs
#     # the final throughput should be bs/obj_value
#     if minmax_throughputs > max_throughput:
#         max_throughput = minmax_throughputs
#         best_result = result
#         best_chunk = chunk
#         best_bs = bs
#         print("update best result: ", max_throughput, best_chunk, best_bs)
L, N, BITs, M_d, M, l, omega, comm, price, theta, gamma = prepare_for_ilp(num_hidden_layers, D, available_bits)
result, obj_value = solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta)
# result, obj_value, device_used = solve_ilp_pulp_with_price(L, N, BITs, M, M_d, l, omega, comm, price, theta, gamma)
result = interpret_ilp_result_i_j_b(result, available_bits)
# store result
result_file_name = 'qpipe_ilp_result.pkl'
root_path = './baseline_result'
import os 
result_path = os.path.join(root_path, result_file_name)
with open(result_path, 'wb') as f:
    pickle.dump(result, f)