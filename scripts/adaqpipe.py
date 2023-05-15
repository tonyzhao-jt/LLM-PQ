# qllm libs
from qllm.models.OPT.opt import model_cards
# qpipe libs
import qpipe
from qpipe.partitioner.indicator import (
    assign_omega_uniform
)
from qpipe.partitioner.utils import (
    interpret_ilp_result_i_j_b
)
from qpipe.cost_model import (
    estimate_single_layer_mem,
    get_mem_with_layer_bit_pair
)
from qpipe.cost_model import price as price_model
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


# import logger
import logging
logger = logging.getLogger(__name__)

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

unit = qpipe._globals.MEM_UNIT
time_mult_times = qpipe._globals.TIME_MULT_TIMES
slo_rate = qpipe._globals.SLO_RATE

# model size
model_size = args.model_size # '66b'
device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
device_numbers = args.device_numbers # [2, 3]
slo_aware = args.SLO_aware
assert len(device_names) == len(device_numbers), "device_names and device_numbers should have the same length"


def solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta, SLO_lat=None):
    prob = pulp.LpProblem("max Latency Minimization Problem", pulp.LpMinimize)
    l_prefill, l_decode = l
    comm_prefill, comm_decode = comm
    # Create a new PuLP model
    B = len(BITs)
    z = pulp.LpVariable.dicts("z", [(i, j, b) for i in range(L) for j in range(N) for b in range(B)], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i, b) for i in range(L) for b in range(B)], cat=pulp.LpBinary)
    T_prefill = pulp.LpVariable.dicts("T_prefill", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous)
    T_decode = pulp.LpVariable.dicts("T_decode", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous)
    T_max = pulp.LpVariable("LAT_max", lowBound=0, cat=pulp.LpContinuous)

    # Define the objective function
    prob += T_max + theta * pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)])

    # Define the constraints
    for i in range(L):
        prob += pulp.lpSum([z[(i, j, b)] for j in range(N) for b in range(B)]) == 1
    for i in range(L):
        for b in range(B):
            prob += pulp.lpSum([z[(i, j, b)] for j in range(N)]) == y[(i, b)]
    for j in range(N):
        prob += pulp.lpSum([z[(i, j, b)] * M[i][b] for i in range(L) for b in range(B)]) <= M_d[j]
        prob += pulp.lpSum([z[(i, j, b)] * l_prefill[i][j][b] for i in range(L) for b in range(B)]) <= T_prefill[j]
        prob += pulp.lpSum([z[(i, j, b)] * l_decode[i][j][b] for i in range(L) for b in range(B)]) <= T_decode[j]
        prob += T_prefill[j] >= comm_prefill[j]
        prob += T_decode[j] >= comm_decode[j]

        prob += T_max >= T_prefill[j]
        prob += T_max >= mu_n * T_decode[j]
    
    # add constraint of SLORate if exits
    if SLO_lat is not None:
        prob += T_max <= SLO_lat
    
    # Solve the problem
    # prob.solve(pulp.apis.PULP_CBC_CMD())
    solver = pulp.GUROBI(msg=True, threads=0, MIPGap=0.005)
    # solver = pulp.GUROBI()
    # solver = pulp.GUROBI(msg=True)
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


def get_latency_with_layer_device_bit_pair(current_D, bit_pairs, lat_cost_model, b, i):
    device_names = list(current_D.values())
    dtypes = set(device_names)
    device_bit_res = {}

    for device_name in dtypes:
        for idx, bit_pair in enumerate(bit_pairs):
            attn_bit, ffn_bit = bit_pair
            device_bit_res[(device_name, bit_pair)] = 0
            if not use_profiler_prediction:
                attn_lat = lat_cost_model.predict_by_model_with_b_i_bit(device_name, 0, b, i, attn_bit)
                ffn_lat = lat_cost_model.predict_by_model_with_b_i_bit(device_name, 1, b, i, ffn_bit)
            else:
                attn_lat = lat_cost_model.predict_by_profiled_with_b_i_bit(device_name, 0, b, i, attn_bit)
                ffn_lat = lat_cost_model.predict_by_profiled_with_b_i_bit(device_name, 1, b, i, ffn_bit)
            if attn_lat is None: # bit is not available
                attn_lat = 9999 # a large number
                # print(device_name, 0, attn_bit)
            if ffn_lat is None:
                ffn_lat = 9999
                # print(device_name, 1, ffn_bit)
            lat = attn_lat + ffn_lat
            device_bit_res[(device_name, bit_pair)] = lat
    # create latency matrix
    lat_device_bits_matrix = np.zeros((len(current_D), len(bit_pairs)))
    for i, device_name in enumerate(device_names):
        for j, bit_pair in enumerate(bit_pairs):
            lat_device_bits_matrix[i, j] = device_bit_res[(device_name, bit_pair)]
    return lat_device_bits_matrix

def get_comm(current_D, comm_cost_model, comm_size):
    device_length = len(current_D)
    comm = np.zeros(device_length)
    for idx in range(device_length):
        comm[idx] = comm_cost_model.predict_comm_time(idx, (idx + 1) % device_length, comm_size)
    return comm


def prepare_for_ilp(num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack, comm_size):
    model_mem_estimator, comm_cost_model, lat_cost_model = cost_model_pack
    global_bz, prefill_bz, bz_decode_max = bz_pack

    L = num_hidden_layers # in partition, regard as a whole
    N = len(current_D) # number of devices

    available_bits = list(set(available_bits))
    BITs = [
        (i, j) for i in available_bits for j in available_bits
    ]
    '''
        Constraint related
    '''
    # device memory
    M_d = np.array([get_single_device_mem_constraints(device_name) for d_rank, device_name in current_D.items()]) 
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
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size_with_bz(prefill_bz, bz_decode_max, unit='MB')[0] 
    M_d[0] -= post_pre_mem
    M_d -= time_mult_times * temp_tensor_mem # torch may not release the tensor immediately, modify the 2 to ensure won't oom
    M_d = np.floor(M_d).astype(int) # floor
    M = np.ceil(M).astype(int) # ceil

    # latency table for prefill and decode
    # latency
    l_prefill = np.zeros((L, N, len(BITs)))
    l_decode = np.zeros((L, N, len(BITs)))

    for i in range(L):
        l_prefill[i, :, :] = get_latency_with_layer_device_bit_pair(current_D, BITs, lat_cost_model, prefill_bz, s)
        l_decode[i, :, :] = get_latency_with_layer_device_bit_pair(current_D, BITs, lat_cost_model, bz_decode_max, mu_n)
    
    # omega
    omega = assign_omega_uniform(L, BITs)

    # comm
    comm_prefill = get_comm(current_D, comm_cost_model, comm_size * s)
    comm_decode = get_comm(current_D, comm_cost_model, comm_size)

    return L, N, BITs, M_d, M, (l_prefill, l_decode), omega, (comm_prefill, comm_decode)

# algo 2
import itertools
from qpipe.utils import partition_a_into_b_bins
from collections import defaultdict
def rename_device_name(D):
    ref_D = {}
    cnt = defaultdict(int)
    for rank, device_name in D.items():
        cnt[device_name] += 1
        ref_D[device_name + str(cnt[device_name])] = rank
    return ref_D
def reset_device_rank_index(D, current_D):
    # D is the previous rank index
    ref_D = rename_device_name(D)
    ref_current_D = rename_device_name(current_D)
    # get mappings
    maps = {}
    for ref_current_device_name, ref_current_rank in ref_current_D.items():
        original_rank = ref_D[ref_current_device_name]
        maps[ref_current_rank] = original_rank
    return maps

# Algo1
def solve_ilp_for_best(T, current_D, comm_size, cost_model_pack, bz_pack):
    num_hidden_layers = len(T) // 2
    SLO_lat = None
    L, N, BITs, M_d, M, l, omega, comm = prepare_for_ilp(num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack, comm_size)
    result, obj_value = solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta, SLO_lat=SLO_lat)
    # result, obj_value, device_used = solve_ilp_pulp_with_price(L, N, BITs, M, M_d, l, omega, comm, price, theta, gamma)
    result = interpret_ilp_result_i_j_b(result, available_bits)
    ilp_res = {}
    ilp_res['obj'] = obj_value
    ilp_res['plan'] = result
    return ilp_res

# Algo2
# enumerate all devices combinations
# enumerata all hybrid micro-batch combinations
# micro-batch candidates
def enumerate_best_result():

    model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names, cost_model_store_path, \
                                                                                                     global_bz, micro_bz, s, n, \
                                                                                                  comm_cost_model_folder=comm_cost_model_dir)
    lat_cost_model.update_profiled_result('/workspace/qpipe/scripts/lat_profiled_result')
    if not use_profiler_prediction:
        lat_cost_model.load_regression_cost_model()

    cost_model_pack = (model_mem_estimator, comm_cost_model, lat_cost_model)
    best_plan = {}
    best_plan['obj'] = 9999999

    num_device_all = sum(device_numbers)
    bz_decode_max = max(partition_a_into_b_bins(global_bz, num_device_all))
    candidate_prefill_bzs = [i for i in range(1, bz_decode_max)]
    # device order candidates
    device_name_with_its_number = list(zip(device_names, device_numbers))
    permutations = itertools.permutations(device_name_with_its_number)
    for perm in permutations:
        new_device_names, new_device_numbers = zip(*perm)
        for prefill_bz in candidate_prefill_bzs:
            current_strategy = prefill_bz, bz_decode_max, new_device_names, new_device_numbers
            # reset the mapping of communication cost model
            current_D = create_device_mesh_and_mem(new_device_names, new_device_numbers)[0]
            maps = reset_device_rank_index(D, current_D)
            comm_cost_model.set_device_rank_map(maps)
            # test 
            bz_pack = (global_bz, prefill_bz, bz_decode_max)
            res = solve_ilp_for_best(T, current_D, comm_size, cost_model_pack, bz_pack)
            if res['obj'] < best_plan['obj']:
                best_plan = res
                best_plan['strategy'] = {
                    'prefill_bz': prefill_bz,
                    'bz_decode_max': bz_decode_max,
                    'device_names': new_device_names,
                    'device_numbers': new_device_numbers,
                    'plan': res['plan']
                }
            comm_cost_model.clear_device_rank_map()
    # log the best plan
    print('best plan: ', best_plan)
    # save the result
    if slo_aware:
        file_name = f'adaqpipe_slo_' + model_size + '_' + device_info + '.pkl'
    else:
        file_name = f'adaqpipe_' + model_size + '_' + device_info + '.pkl'
    folder = '/workspace/qpipe/scripts/strategy'
    save_with_pickle(best_plan, file_name, folder)
    return best_plan
    

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
# some hyper parameters
gamma = 0.5 # expected generated tokens
theta = 0.1 # control the concern for accuracy
mu_n = int(gamma * n)
available_bits = [2, 3, 4, 8,  '8:tc', '8:tc-li', 16] # we now can do hardware-aware quantization with 8:tc and 8:tc-li
D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
# max_device_mem can be used to check whether OOM or not
use_profiler_prediction = False # use regression model to predict or load predictor
# target model configuration
device_info = get_device_info(device_names, device_numbers)
comm_cost_model_dir = f'/workspace/qpipe/scripts/comm_cost_model/{device_info}'
cost_model_store_path = None # initialize the cost model
enumerate_best_result()