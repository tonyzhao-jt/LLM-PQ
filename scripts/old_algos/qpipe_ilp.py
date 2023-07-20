# qllm libs
from qllm.models.OPT.opt import model_cards
# qpipe libs
import shaq
from shaq.partitioner.indicator import (
    assign_omega_uniform
)
from shaq.partitioner.utils import (
    interpret_ilp_result_i_j_b
)
from shaq.cost_model import (
    estimate_single_layer_mem,
    get_mem_with_layer_bit_pair
)
from shaq.cost_model import price as price_model
from shaq.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    get_slo
)

from shaq.utils import (
    save_with_pickle
)

# default libs
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
parser.add_argument('--SLO-aware',  action='store_true', help='add slo into constraints')
args = parser.parse_args()

unit = shaq._globals.MEM_UNIT
time_mult_times = shaq._globals.TIME_MULT_TIMES
slo_rate = shaq._globals.SLO_RATE

# model size
model_size = args.model_size # '66b'
device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
device_numbers = args.device_numbers # [2, 3]
slo_aware = args.SLO_aware
assert len(device_names) == len(device_numbers), "device_names and device_numbers should have the same length"


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



def solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta, SLO_lat=None):
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
        prob += LAT_max >= LAT[j]
        prob += LAT_max >= comm[j]
    
    # add constraint of SLORate if exits
    if SLO_lat is not None:
        prob += LAT_max <= SLO_lat
    
    # Solve the problem
    # prob.solve(pulp.apis.PULP_CBC_CMD())
    # solver = pulp.GUROBI(msg=True, threads=0, timeLimit=100, MIPGap=0.003)
    # solver = pulp.GUROBI()
    solver = pulp.GUROBI(msg=True)
    prob.solve(solver)

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
    
    mem_bits_vector = get_mem_with_layer_bit_pair(BITs, model_mem_estimator)
    M = np.tile(mem_bits_vector, (L, 1))

    # reduce the embedding size on device 0 for M_d
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size(unit='MB')[0] / chunk_size
    temp_emb_mem = model_mem_estimator.calculate_temp_embedding_tensor_size(unit='MB')[0]
    M_d[0] -= post_pre_mem
    M_d -= time_mult_times * temp_tensor_mem # torch may not release the tensor immediately, modify the 2 to ensure won't oom
    M_d = np.floor(M_d).astype(int) # floor
    M = np.ceil(M).astype(int) # ceil

    # latency
    l = np.zeros((L, N, len(BITs)))
    lat_device_bits_matrix = get_latency_with_layer_device_bit_pair(D, BITs)
    for i in range(L):
        l[i, :, :] = lat_device_bits_matrix
    
    # omega
    omega = assign_omega_uniform(L, BITs)

    # comm
    comm = get_comm(D)

    # control the concern for latency
    theta = 0.1

    # price related
    gamma = 0.01
    # get price
    price = np.zeros(N)
    for i in range(N):
        price[i] = price_model.get_price(D[i])
    return L, N, BITs, M_d, M, l, omega, comm, price, theta, gamma

'''
    Initiailization
'''
from shaq.partitioner import gen_config
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

available_bits = [2, 3, 4, 8, '8:tc-li', 16] # we now can do hardware-aware quantization with 8:tc and 8:tc-li

if slo_aware:
    SLO_lat = get_slo(model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size, \
                device_names, use_profiler_prediction=True, verbose=True)
else:
    SLO_lat = None

L, N, BITs, M_d, M, l, omega, comm, price, theta, gamma = prepare_for_ilp(num_hidden_layers, D, available_bits)

result, obj_value = solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta, SLO_lat=SLO_lat)
# result, obj_value, device_used = solve_ilp_pulp_with_price(L, N, BITs, M, M_d, l, omega, comm, price, theta, gamma)
result = interpret_ilp_result_i_j_b(result, available_bits)

# save the result
if slo_aware:
    file_name = f'adaqpipe_slo_' + model_size + '_' + device_info + '.pkl'
else:
    file_name = f'adaqpipe_' + model_size + '_' + device_info + '.pkl'
folder = '/workspace/qpipe/scripts/strategy'
save_with_pickle(result, file_name, folder)