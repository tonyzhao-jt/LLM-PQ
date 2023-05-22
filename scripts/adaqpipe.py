# qllm libs
from qllm.models.OPT.opt import model_cards
# qpipe libs
import qpipe
from qpipe.partitioner.indicator import (
    assign_omega_uniform
)
from qpipe.partitioner.utils import (
    interpret_ilp_result_i_j_b,
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
    get_slo,
    force_zero,
    decouple_result_group,
    get_latency_with_layer_device_bit_pair
)

from qpipe.utils import (
    get_default_decode_bz,
    save_with_pickle, get_available_bits_pair
)

import pickle

# import logger
import logging
logger = logging.getLogger(__name__)

# default libs
import numpy as np 
# setup ilp configs
import pulp
import gurobipy as gp
from utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE
)
unit = qpipe._globals.MEM_UNIT
time_mult_times = qpipe._globals.TIME_MULT_TIMES
slo_rate = qpipe._globals.SLO_RATE

import math 

def solve_ilp_pulp(L, N, BITs, M, M_d, l, omega, comm, theta, bz_pack):
    prob = pulp.LpProblem("max Latency Minimization Problem", pulp.LpMinimize)
    l_prefill, l_decode = l
    comm_prefill, comm_decode = comm
    global_bz, prefill_bz, bz_decode_max = bz_pack
    # Create a new PuLP model
    B = len(BITs)
    z = pulp.LpVariable.dicts("z", [(i, j, b) for i in range(L) for j in range(N) for b in range(B)], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i, b) for i in range(L) for b in range(B)], cat=pulp.LpBinary)
    T_prefill_j = pulp.LpVariable.dicts("T_prefill_j", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous)
    T_decode_j = pulp.LpVariable.dicts("T_decode_j", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous)
    T_prefill = pulp.LpVariable("T_prefill", lowBound=0, cat=pulp.LpContinuous)
    T_decode = pulp.LpVariable("T_decode", lowBound=0, cat=pulp.LpContinuous)
    T_sum = pulp.LpVariable("T_sum", lowBound=0, cat=pulp.LpContinuous)

    # Define the objective function
    prob += T_sum + theta * pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)])

    force_zero(l_prefill, z, prob)
    # Define the constraints
    for i in range(L):
        prob += pulp.lpSum([z[(i, j, b)] for j in range(N) for b in range(B)]) == 1
    for i in range(L):
        for b in range(B):
            prob += pulp.lpSum([z[(i, j, b)] for j in range(N)]) == y[(i, b)]
    for j in range(N):
        prob += pulp.lpSum([z[(i, j, b)] * M[i][b] for i in range(L) for b in range(B)]) <= M_d[j]
        prob += pulp.lpSum([z[(i, j, b)] * l_prefill[i][j][b] for i in range(L) for b in range(B)]) <= T_prefill_j[j]
        prob += pulp.lpSum([z[(i, j, b)] * l_decode[i][j][b] for i in range(L) for b in range(B)]) <= T_decode_j[j]

        prob += T_prefill_j[j] >= comm_prefill[j] * s 
        prob += T_decode_j[j] >= comm_decode[j]
        prob += T_prefill >= T_prefill_j[j]
        prob += T_decode >= T_decode_j[j]


    T_sum = (math.ceil(global_bz / prefill_bz) + 1) * T_prefill + \
          (math.ceil(global_bz / bz_decode_max) + 1) * T_decode * (mu_n - 1)
    
    # Solve the problem
    # prob.solve(pulp.apis.PULP_CBC_CMD())
    solver = pulp.GUROBI(msg=False)
    # solver = pulp.GUROBI()
    # solver = pulp.GUROBI(msg=True)
    status = prob.solve(solver)

    if status == pulp.LpStatusOptimal:
        # Print the solution status
        print("Adaqpipe Result Found")
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
    else:
        return NOT_AVAILABLE, 1e10

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
    assert L % group_size == 0, f'L should be divisible by group_size, but L={L}, group_size={group_size}'
    group_L = L // group_size

    BITs = get_available_bits_pair(available_bits)
    '''
        Constraint related
    '''
    M_d = np.array([get_single_device_mem_constraints(device_name) for d_rank, device_name in D.items()]) 
    mem_bits_vector = get_mem_with_layer_bit_pair(BITs, model_mem_estimator)
    M = np.tile(mem_bits_vector, (group_L, 1)) * group_size # repeat the mem_bits_vector for group_L times

    # reduce the embedding size on device 0
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size_with_bz(prefill_bz, bz_decode_max, unit='MB')[0] 
    M_d[0] -= post_pre_mem
    M_d -= time_mult_times * temp_tensor_mem # torch may not release the tensor immediately, modify the 2 to ensure won't oom
    M_d = np.floor(M_d).astype(int) # floor
    M = np.ceil(M).astype(int) # ceil

    # latency table for prefill and decode
    # latency
    l_prefill = np.zeros((group_L, N, len(BITs)))
    l_decode = np.zeros((group_L, N, len(BITs))) 

    for i in range(group_L):
        l_prefill[i, :, :] = get_latency_with_layer_device_bit_pair(current_D, BITs, lat_cost_model, prefill_bz, s, 0, \
                                                                     use_profiler_prediction=use_profiler_prediction) * group_size
        l_decode[i, :, :] = get_latency_with_layer_device_bit_pair(current_D, BITs, lat_cost_model, bz_decode_max, 1, s + int(mu_n / 2), \
                                                                   use_profiler_prediction=use_profiler_prediction) * group_size
    
    # omega
    omega = assign_omega_uniform(group_L, BITs)
    if omega_file is not None:
        # open and load with pickle
        with open(omega_file, 'rb') as f:
            omega_loaded = pickle.load(f)
        # check whether the shape is matched, as raise error
        # all_BITs = get_available_bits_pair(qpipe._globals.AVAILABLE_BITS)
        # BITs_idx = [all_BITs.index(bit_pair) for bit_pair in BITs]
        # omega_loaded = omega_loaded[:, BITs_idx]
        # if omega_loaded.shape[0] != group_L and omega_loaded.shape[0] == L:
        #     new_omega_loaded = np.zeros((group_L, omega_loaded.shape[1]))
        #     for i in range(group_L):
        #         new_omega_loaded[i] = np.mean(omega_loaded[i*group_size:(i+1)*group_size], axis=0)
        #     omega_loaded = new_omega_loaded
        
        if omega_loaded.shape != omega.shape:
            print(omega_loaded.shape, omega.shape)
            raise ValueError('omega shape mismatched')
        omega = omega_loaded

    # comm
    comm_prefill = get_comm(current_D, comm_cost_model, comm_size * s)
    comm_decode = get_comm(current_D, comm_cost_model, comm_size)

    return group_L, N, BITs, M_d, M, (l_prefill, l_decode), omega, (comm_prefill, comm_decode)

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

# for debug
def check_performance(lat, result):
    # check lat
    for layer_idx, (device, bit_idx) in result.items():
        cur_lat = lat[(layer_idx, device, bit_idx)]
        print('layer {}, device {}, bit {}, lat {}'.format(layer_idx, device, bit_idx, cur_lat))
    
# Algo1
def solve_ilp_for_best(T, current_D, comm_size, cost_model_pack, bz_pack):
    num_hidden_layers = len(T) // 2
    SLO_lat = None
    L, N, BITs, M_d, M, lat, omega, comm = prepare_for_ilp(num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack, comm_size)
    result, obj_value = solve_ilp_pulp(L, N, BITs, M, M_d, lat, omega, comm, theta, bz_pack)
    # result, obj_value, device_used = solve_ilp_pulp_with_price(L, N, BITs, M, M_d, l, omega, comm, price, theta, gamma)
    # check_performance(lat[0], result)
    if result != NOT_AVAILABLE:
        result = decouple_result_group(group_size, result)
        result = interpret_ilp_result_i_j_b(result, BITs)
    ilp_res = {}
    ilp_res['obj'] = obj_value
    ilp_res['plan'] = result
    return ilp_res

# Algo2
# enumerate all devices combinations
# enumerata all hybrid micro-batch combinations
# micro-batch candidates
def enumerate_best_result(args):
    model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = args.init_pack 
    cost_model_pack = (model_mem_estimator, comm_cost_model, lat_cost_model)
    best_plan = {}
    best_plan['obj'] = 9999999

    num_device_all = sum(device_numbers)
    strat = partition_a_into_b_bins(global_bz, num_device_all)
    bz_decode_max = get_default_decode_bz(global_bz, num_device_all)
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
                best_plan = {
                    'prefill_bz': prefill_bz,
                    'bz_decode_max': bz_decode_max,
                    'bz_decode_bss': strat,
                    'device_names': new_device_names,
                    'device_numbers': new_device_numbers,
                    'plan': res['plan'],
                    'obj': res['obj'],
                    'D': current_D,
                    'maps': maps,
                }
            comm_cost_model.clear_device_rank_map()
    # log the best plan
    print('best plan: ', best_plan)
    # save the result
    # file_name = f'adaqpipe_' + model_size + '_' + device_info + '.pkl'
    # folder = '/workspace/qpipe/scripts/strategy'
    # save_with_pickle(best_plan, file_name, folder)
    return best_plan

'''
    Initiailization
'''
from qpipe.partitioner import gen_config
# global variables
global_bz, micro_bz = None, None
s, n = None, None
model_size, device_info = None, None
use_profiler_prediction = False
D = None
device_numbers, device_names = None, None
available_bits = None
config = None
theta = None
mu_n = None
cost_model_store_path = '/workspace/qpipe/scripts/cost_model_store'
comm_cost_model_dir = '/workspace/qpipe/scripts/comm_cost_model'
lat_profile_result_path = '/workspace/qpipe/scripts/lat_profiled_result'
omega_file = None
ilp_seed = 0
group_size = 1
def main(args):
    global global_bz, micro_bz, s, n
    global model_size, device_info
    global use_profiler_prediction
    global D 
    global device_numbers, device_names
    global available_bits
    global config
    global theta
    global mu_n
    global cost_model_store_path, comm_cost_model_dir
    global lat_profile_result_path
    global omega_file
    global ilp_seed
    global group_size
    # global variables

    omega_file = args.omega_file
    ilp_seed = args.ilp_seed
    group_size = args.group_size
    ilp_tolerance = args.ilp_tolerance
    if ilp_tolerance is not None:
        pulp.LpSolverDefault.eps = ilp_tolerance

    global_bz = gen_config.global_bz
    micro_bz = gen_config.micro_bz
    s = gen_config.s
    n = gen_config.n
    model_size = args.model_size # '66b'
    device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers # [2, 3]
    # generation configs
    config = args.config

    gamma = qpipe._globals.gamma # expected generated tokens
    theta = qpipe._globals.theta # control the concern for accuracy
    mu_n = int(gamma * n)
    available_bits = qpipe._globals.AVAILABLE_BITS # we now can do hardware-aware quantization with 8:tc and 8:tc-li
    D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
    # max_device_mem can be used to check whether OOM or not
    use_profiler_prediction = args.use_profiler_prediction # use regression model to predict or load predictor
    # target model configuration
    device_info = get_device_info(device_names, device_numbers)
    comm_cost_model_dir = f'{args.comm_cost_model_dir}/{device_info}'
    cost_model_store_path = None # initialize the cost model

    lat_profile_result_path = args.lat_profile_dir
    return enumerate_best_result(args)

if __name__ == '__main__':
    ilp_env()
    args = common_argparser()
    main(args)
    # model size
