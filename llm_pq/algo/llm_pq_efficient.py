'''
    Efficient Version of llm_pq
    - first group the device of the same type
'''

from .. import _globals
from ..partitioner.indicator import (
    assign_omega_uniform, assign_omega_constant
)
from ..partitioner.utils import (
    interpret_ilp_result_i_j_b,
)
from ..cost_model import (
    get_mem_with_layer_bit_pair
)
from ..partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    get_slo,
    force_zero,
    decouple_result_group,
    get_latency_with_layer_device_bit_pair
)

from ..utils import (
    get_default_decode_bz,
    save_with_pickle, get_available_bits_pair,
    get_factors,
    convert_D_to_ranked_device
)

import pickle
import os 
# import logger
import logging
logger = logging.getLogger(__name__)

# default libs
import numpy as np 
# setup ilp configs
import pulp
import gurobipy as gp
from .utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE,
    create_ilp_solver
)
from .mem_utils import get_device_topo_available_mem_with_order, get_M_with_bitwidth_pair
unit = _globals.MEM_UNIT
slo_rate = _globals.SLO_RATE

import math 


def solve_ilp_pulp_grouped_device(L, N, BITs, M, M_d, l, omega, comm, theta, bz_pack, group_D_nums):
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

    # from alpa
    # cost = stage_sum + (micro_bs - 1) * lowest_stage_cost
    prefill_micro_bs_num = math.ceil(global_bz / prefill_bz)
    decode_micro_bs_num = math.ceil(global_bz / bz_decode_max)
    prefill_stage_sum = pulp.lpSum([T_prefill_j[j] for j in range(N)])
    decode_stage_sum = pulp.lpSum([T_decode_j[j] for j in range(N)])
    prob += prefill_stage_sum + (prefill_micro_bs_num - 1) * T_prefill + \
          decode_stage_sum + (decode_micro_bs_num - 1) * T_decode * (mu_n - 1) \
          + theta * pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)])

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
        prob += T_prefill_j[j] >= comm_prefill[j] 
        prob += T_prefill >= (1 / group_D_nums[j]) * T_prefill_j[j] 

        prob += pulp.lpSum([z[(i, j, b)] * l_decode[i][j][b] for i in range(L) for b in range(B)]) <= T_decode_j[j]
        prob += T_decode_j[j] >= comm_decode[j]
        prob += T_decode >= (1 / group_D_nums[j]) * T_decode_j[j] 

    solver = create_ilp_solver(verbose_ilp, ilp_time_limit, ilp_tolerance)
    status = prob.solve(solver)

    if status == pulp.LpStatusOptimal:
        # Print the solution status
        # print("llm_pq Result Found")
        # store the optimal solution
        result = {}
        # print z variable result
        for i in range(L):
            for j in range(N):
                for b in range(B):
                    if z[(i, j, b)].varValue > 0:
                        # print("z[{}, {}, {}] = {}".format(i, j, b, z[(i, j, b)].varValue))
                        result[i] = (j, b)
        # min([v.varValue for k, v in z.items()])
        # sum omega
        # print("omega = {}".format(pulp.value(pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)]))))
        # print latency T_sum
        T_sum = pulp.value(prefill_stage_sum) + (prefill_micro_bs_num - 1) * T_prefill.varValue + \
          pulp.value(decode_stage_sum) + (decode_micro_bs_num - 1) * T_decode.varValue * (mu_n - 1)
        print("T_sum = {}".format(T_sum))
        print("Objective = {}".format(pulp.value(prob.objective)))
        return result, pulp.value(prob.objective)
    else:
        return NOT_AVAILABLE, 1e10

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

    # from alpa
    # cost = stage_sum + (micro_bs - 1) * lowest_stage_cost
    prefill_micro_bs_num = math.ceil(global_bz / prefill_bz)
    decode_micro_bs_num = math.ceil(global_bz / bz_decode_max)
    prefill_stage_sum = pulp.lpSum([T_prefill_j[j] for j in range(N)])
    decode_stage_sum = pulp.lpSum([T_decode_j[j] for j in range(N)])
    prob += prefill_stage_sum + (prefill_micro_bs_num - 1) * T_prefill + \
          decode_stage_sum + (decode_micro_bs_num - 1) * T_decode * (mu_n - 1) \
          + theta * pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)])

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

        prob += T_prefill_j[j] >= comm_prefill[j] 
        prob += T_decode_j[j] >= comm_decode[j]
        prob += T_prefill >= T_prefill_j[j]
        prob += T_decode >= T_decode_j[j]

    solver = create_ilp_solver(verbose_ilp, ilp_time_limit, ilp_tolerance)
    status = prob.solve(solver)

    if status == pulp.LpStatusOptimal:
        # Print the solution status
        # print("llm_pq Result Found")
        # store the optimal solution
        result = {}
        # print z variable result
        for i in range(L):
            for j in range(N):
                for b in range(B):
                    if z[(i, j, b)].varValue > 0:
                        # print("z[{}, {}, {}] = {}".format(i, j, b, z[(i, j, b)].varValue))
                        result[i] = (j, b)
        # min([v.varValue for k, v in z.items()])
        # sum omega
        # print("omega = {}".format(pulp.value(pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)]))))
        # print latency T_sum
        T_sum = pulp.value(prefill_stage_sum) + (prefill_micro_bs_num - 1) * T_prefill.varValue + \
          pulp.value(decode_stage_sum) + (decode_micro_bs_num - 1) * T_decode.varValue * (mu_n - 1)
        print("T_sum = {}".format(T_sum))
        print("Objective = {}".format(pulp.value(prob.objective)))
        return result, pulp.value(prob.objective)
    else:
        return NOT_AVAILABLE, 1e10

def get_comm(current_D, comm_cost_model, comm_size):
    device_length = len(current_D)
    comm = np.zeros(device_length)
    for idx in range(device_length):
        comm[idx] = comm_cost_model.predict_comm_time(idx, (idx + 1) % device_length, comm_size)
    return comm


def prepare_for_ilp_grouped_device(num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack):
    time_mult_times = _globals.TIME_MULT_TIMES
    model_mem_estimator, comm_cost_model, lat_cost_model = cost_model_pack
    global_bz, prefill_bz, bz_decode_max = bz_pack

    L = num_hidden_layers # in partition, regard as a whole

    assert L % group_size == 0, f'L should be divisible by group_size, but L={L}, group_size={group_size}'
    group_L = L // group_size

    BITs = get_available_bits_pair(available_bits)
    '''
        Constraint related
    '''
    # get memory requirement of each layer
    M = get_M_with_bitwidth_pair(BITs, model_mem_estimator, group_L, group_size)
    M_d = get_device_topo_available_mem_with_order(current_D, model_mem_estimator, prefill_bz, bz_decode_max, time_mult_times=time_mult_times)

    '''
        Change it to device grouped version
        e.g. original: {0: 'Tesla_V100-SXM2-32GB', 1: 'Tesla_V100-SXM2-32GB', 2: 'NVIDIA_A100-SXM4-40GB', 3: 'NVIDIA_A100-SXM4-40GB'}
        grouped: {0: 'Tesla_V100-SXM2-32GB', 1: 'NVIDIA_A100-SXM4-40GB'}
    '''
    device_rank_list, _ = convert_D_to_ranked_device(current_D)
    N = len(device_rank_list) # number of device types
    idx = 0
    grouped_D = {}
    for device_name in device_rank_list.keys():
        grouped_D[idx] = device_name
        idx += 1
    idx = 0
    md_idx = 0
    grouped_M_d = np.zeros(N)
    for device_name, num in device_rank_list.items():
        grouped_M_d[idx] = sum([M_d[i] for i in range(md_idx, md_idx + num)])
        md_idx += num
        idx += 1

    # latency table for prefill and decode
    l_prefill = np.zeros((group_L, N, len(BITs)))
    l_decode = np.zeros((group_L, N, len(BITs))) 

    for i in range(group_L):
        l_prefill[i, :, :] = get_latency_with_layer_device_bit_pair(grouped_D, BITs, lat_cost_model, prefill_bz, s, 0, \
                                                                     use_profiler_prediction=use_profiler_prediction) * group_size
        l_decode[i, :, :] = get_latency_with_layer_device_bit_pair(grouped_D, BITs, lat_cost_model, bz_decode_max, 1, s + int(mu_n / 2), \
                                                                   use_profiler_prediction=use_profiler_prediction) * group_size
    # preposet
    if len(grouped_D) > 1:
        first_device_name = grouped_D[0]
        # prefill
        prefill_prepost_cost = lat_cost_model.fetch_prepost_lat(first_device_name, 0, prefill_bz, s) * math.ceil(global_bz / prefill_bz)
        # decode
        # print(bz_decode_max, s + int(mu_n / 2))
        decode_prepost_cost = lat_cost_model.fetch_prepost_lat(first_device_name, 1, bz_decode_max, s + int(mu_n / 2))
        # add to l_prefill and l_decode
        l_prefill[:, 0, :] += prefill_prepost_cost
        l_decode[:, 0, :] += decode_prepost_cost
    # omega
    omega = assign_omega_constant(group_L, BITs)
    if omega_file is not None:
        # open and load with pickle
        with open(omega_file, 'rb') as f:
            omega_loaded = pickle.load(f)
        # check whether the shape is matched, as raise error
        # all_BITs = get_available_bits_pair(_globals.AVAILABLE_BITS)
        # BITs_idx = [all_BITs.index(bit_pair) for bit_pair in BITs]
        # omega_loaded = omega_loaded[:, BITs_idx]
        if omega_loaded.shape[0] != group_L and omega_loaded.shape[0] == L:
            new_omega_loaded = np.zeros((group_L, omega_loaded.shape[1]))
            for i in range(group_L):
                new_omega_loaded[i] = np.mean(omega_loaded[i*group_size:(i+1)*group_size], axis=0)
            omega_loaded = new_omega_loaded
        
        if omega_loaded.shape != omega.shape:
            print(omega_loaded.shape, omega.shape)
            raise ValueError('omega shape mismatched')
        omega = omega_loaded

    # comm
    comm_prefill = (model_mem_estimator.h1 * prefill_bz * s) * 2 / 1024 / 1024
    comm_prefill = get_comm(current_D, comm_cost_model, comm_prefill) * comm_multiplier

    comm_decode = (model_mem_estimator.h1 * bz_decode_max * 1) * 2 / 1024 / 1024
    comm_decode = get_comm(current_D, comm_cost_model, comm_decode) * comm_multiplier

    # interested per-grouped device latency
    comm_idx_interested = []
    idx = 0
    for device_name, num in device_rank_list.items():
        comm_idx_interested.append(idx + num - 1)
        idx += num
    comm_prefill_interested = comm_prefill[comm_idx_interested]
    comm_decode_interested = comm_decode[comm_idx_interested]
    group_D_nums = {}
    idx = 0
    for device_name, num in device_rank_list.items():
        group_D_nums[idx] = num
        idx += 1
    # print('----- communication cost ---- ')
    # print(comm_prefill, comm_decode)
    return group_L, N, BITs, grouped_M_d, M, (l_prefill, l_decode), omega, (comm_prefill_interested, comm_decode_interested), (group_D_nums, device_rank_list)


def prepare_for_ilp(num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack):
    time_mult_times = _globals.TIME_MULT_TIMES
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
    # get memory requirement of each layer
    M = get_M_with_bitwidth_pair(BITs, model_mem_estimator, group_L, group_size)
    M_d = get_device_topo_available_mem_with_order(current_D, model_mem_estimator, prefill_bz, bz_decode_max, time_mult_times=time_mult_times)

    # latency table for prefill and decode
    # latency
    l_prefill = np.zeros((group_L, N, len(BITs)))
    l_decode = np.zeros((group_L, N, len(BITs))) 

    for i in range(group_L):
        l_prefill[i, :, :] = get_latency_with_layer_device_bit_pair(current_D, BITs, lat_cost_model, prefill_bz, s, 0, \
                                                                     use_profiler_prediction=use_profiler_prediction) * group_size
        l_decode[i, :, :] = get_latency_with_layer_device_bit_pair(current_D, BITs, lat_cost_model, bz_decode_max, 1, s + int(mu_n / 2), \
                                                                   use_profiler_prediction=use_profiler_prediction) * group_size
    # preposet
    if len(current_D) > 1:
        first_device_name = current_D[0]
        # prefill
        prefill_prepost_cost = lat_cost_model.fetch_prepost_lat(first_device_name, 0, prefill_bz, s) * math.ceil(global_bz / prefill_bz)
        # decode
        # print(bz_decode_max, s + int(mu_n / 2))
        decode_prepost_cost = lat_cost_model.fetch_prepost_lat(first_device_name, 1, bz_decode_max, s + int(mu_n / 2))
        # add to l_prefill and l_decode
        l_prefill[:, 0, :] += prefill_prepost_cost
        l_decode[:, 0, :] += decode_prepost_cost
    # omega
    omega = assign_omega_constant(group_L, BITs)
    if omega_file is not None:
        # open and load with pickle
        with open(omega_file, 'rb') as f:
            omega_loaded = pickle.load(f)
        # check whether the shape is matched, as raise error
        # all_BITs = get_available_bits_pair(_globals.AVAILABLE_BITS)
        # BITs_idx = [all_BITs.index(bit_pair) for bit_pair in BITs]
        # omega_loaded = omega_loaded[:, BITs_idx]
        if omega_loaded.shape[0] != group_L and omega_loaded.shape[0] == L:
            new_omega_loaded = np.zeros((group_L, omega_loaded.shape[1]))
            for i in range(group_L):
                new_omega_loaded[i] = np.mean(omega_loaded[i*group_size:(i+1)*group_size], axis=0)
            omega_loaded = new_omega_loaded
        
        if omega_loaded.shape != omega.shape:
            print(omega_loaded.shape, omega.shape)
            raise ValueError('omega shape mismatched')
        omega = omega_loaded

    # comm
    comm_prefill = (model_mem_estimator.h1 * prefill_bz * s) * 2 / 1024 / 1024
    comm_prefill = get_comm(current_D, comm_cost_model, comm_prefill) * comm_multiplier

    comm_decode = (model_mem_estimator.h1 * bz_decode_max * 1) * 2 / 1024 / 1024
    comm_decode = get_comm(current_D, comm_cost_model, comm_decode) * comm_multiplier

    # print('----- communication cost ---- ')
    # print(comm_prefill, comm_decode)
    return group_L, N, BITs, M_d, M, (l_prefill, l_decode), omega, (comm_prefill, comm_decode)


def incremental_algo():
    return None, None 
# algo 2
import itertools
from ..utils import partition_a_into_b_bins
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
def solve_ilp_for_best(T, current_D, cost_model_pack, bz_pack):
    print("Try", bz_pack)
    num_hidden_layers = len(T) // 2
    SLO_lat = None
    # first step
    # grouped device result
    L, N, BITs, M_d, M, lat, omega, comm, (group_D_nums, device_rank_list) \
          = prepare_for_ilp_grouped_device(num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack)
    result, obj_value = solve_ilp_pulp_grouped_device(L, N, BITs, M, M_d, lat, omega, comm, theta, bz_pack, group_D_nums)
    if result != NOT_AVAILABLE:
        result = decouple_result_group(group_size, result)
        result = interpret_ilp_result_i_j_b(result, BITs)

        # second step
        # based on the group_D_nums, get sub D from the current_D
        sub_D_group = [{} for _ in range(N)]
        idx = 0
        device_idx = 0
        for device_name, num in device_rank_list.items():
            for k in range(device_idx, device_idx+num):
                sub_D_group[idx][k] = device_name
            idx += 1 # N idx
            device_idx += num

        partition_result = result['partition_result']
        bit_assignment = result['bit_assignment']
        if use_incremental:
            # incremental algo
            new_partition_result, new_bit_assignment = incremental_algo()
        else:
            new_partition_result = {}
            new_bit_assignment = {}
            new_obj_value = 0
            # the original ILP 
            L, original_N, BITs, M_d, M, lat, omega, comm = prepare_for_ilp(num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack)
            # run ILP for sub problem again
            for i in range(N):
                assigned_layers = partition_result[i]
                # assigned_bitwidths = [bit_assignment[j] for j in range(assigned_layers[0], assigned_layers[1])]
                sub_L = (assigned_layers[1] - assigned_layers[0]) // group_size
                sub_L_idx_0 = assigned_layers[0] // group_size
                sub_L_idx_1 = assigned_layers[1] // group_size
                sub_N = group_D_nums[i]
                sub_N_index_0 = sum([group_D_nums[j] for j in range(i)])
                sub_N_index_1 = sub_N_index_0 + sub_N
                sub_D = sub_D_group[i]
                sub_M_d = [M_d[j] for j in sub_D.keys()]
                sub_M = M[sub_L_idx_0:sub_L_idx_1, :]
                # groupL, N, bits
                sub_lat = [
                    lat[0][sub_L_idx_0:sub_L_idx_1,sub_N_index_0:sub_N_index_1,:],
                    lat[1][sub_L_idx_0:sub_L_idx_1,sub_N_index_0:sub_N_index_1,:]
                ]
                sub_omega = omega[sub_L_idx_0:sub_L_idx_1, :]
                # this is wrong, but the intra-group communication is not important
                sub_comm = [
                    comm[0][sub_N_index_0:sub_N_index_1],
                    comm[1][sub_N_index_0:sub_N_index_1]
                ]
                sub_result, sub_obj_value = solve_ilp_pulp(sub_L,sub_N, BITs, sub_M, sub_M_d, sub_lat, sub_omega, sub_comm, theta, bz_pack)
                # result, obj_value, device_used = solve_ilp_pulp_with_price(L, N, BITs, M, M_d, l, omega, comm, price, theta, gamma)
                # check_performance(lat[0], result)
                if result != NOT_AVAILABLE:
                    new_obj_value += sub_obj_value
                    sub_result = decouple_result_group(group_size, sub_result)
                    sub_result = interpret_ilp_result_i_j_b(sub_result, BITs)
                    sub_partition_result = sub_result['partition_result']
                    sub_bit_assignment = sub_result['bit_assignment']
                    # copy the result
                    for device_idx, (start_layer, end_layer) in sub_partition_result.items():
                        new_partition_result[sub_N_index_0 + device_idx] = [assigned_layers[0]+ start_layer, assigned_layers[0] + end_layer]
                    for layer_idx, bit in sub_bit_assignment.items():
                        new_bit_assignment[int(assigned_layers[0] * 2) + layer_idx] = bit # *2 because we allow selfatten ffn decompose
                    print("Done subproblem")
                else:
                    raise ValueError('sub problem should not be infeasible, try original llm_pq.')

        # check new result key is consistent to old ones
        assert len(new_partition_result) == len(current_D), f'Wrong in N'
        assert len(new_bit_assignment) == len(bit_assignment), f'new_bit_assignment {new_bit_assignment} is not consistent to old one {bit_assignment}'
        # assign the new result
        result['partition_result'] = new_partition_result
        result['bit_assignment'] = new_bit_assignment    

    ilp_res = {}
    ilp_res['obj'] = obj_value
    ilp_res['plan'] = result
    return ilp_res

# Algo2
# enumerate all devices combinations
# enumerata all hybrid micro-batch combinations
# micro-batch candidates

def enumerate_best_result(args):
    model_mem_estimator, comm_cost_model, lat_cost_model, T = args.init_pack 
    cost_model_pack = (model_mem_estimator, comm_cost_model, lat_cost_model)
    best_plan = {}
    best_plan['obj'] = float("inf")

    num_device_all = sum(device_numbers)
    strat = partition_a_into_b_bins(global_bz, num_device_all)
    bz_decode_max = get_default_decode_bz(global_bz, num_device_all)
    # candidate_prefill_bzs = [i for i in range(1, bz_decode_max + 1)]
    candidate_prefill_bzs = get_factors(bz_decode_max)
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
            res = solve_ilp_for_best(T, current_D, cost_model_pack, bz_pack)
            print(res['obj'], best_plan['obj'])
            if res['obj'] < best_plan['obj']:
                print("Better Plan Generated")
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
    # file_name = f'llm_pq_' + model_size + '_' + device_info + '.pkl'
    # folder = '/workspace/qpipe/scripts/strategy'
    # save_with_pickle(best_plan, file_name, folder)
    return best_plan

'''
    Initiailization
'''
from ..partitioner import gen_config
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
ROOT_DIR=os.environ.get('ROOT_DIR', None)
assert ROOT_DIR is not None, "ROOT_DIR is not set"
cost_model_store_path = f'{ROOT_DIR}/scripts/cost_model_store'
comm_cost_model_dir = f'{ROOT_DIR}/scripts/comm_cost_model'
lat_profile_result_path = f'{ROOT_DIR}/scripts/lat_profiled_result'
omega_file = None
group_size = 1
ilp_seed = 0
ilp_time_limit = 20
ilp_tolerance = None
verbose_ilp = False
comm_multiplier = 1
use_incremental = False
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
    global group_size
    global ilp_seed, ilp_time_limit, ilp_tolerance, verbose_ilp
    global comm_multiplier
    global use_incremental
    # global variables

    omega_file = args.omega_file
    ilp_seed = args.ilp_seed
    group_size = args.group_size
    ilp_tolerance = args.ilp_tolerance
    ilp_time_limit = args.ilp_time_limit
    ilp_tolerance = args.ilp_tolerance
    verbose_ilp = args.debug
    if ilp_tolerance is not None:
        pulp.LpSolverDefault.eps = ilp_tolerance
    comm_multiplier = args.comm_multiplier

    use_incremental = args.llm_pq_ef_incremental

    global_bz = gen_config.global_bz
    micro_bz = gen_config.micro_bz
    s = gen_config.s
    n = gen_config.n
    model_size = args.model_size # '66b'
    device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers # [2, 3]
    # generation configs
    config = args.config

    gamma = _globals.gamma # expected generated tokens
    theta = _globals.theta # control the concern for accuracy
    mu_n = int(gamma * n)
    available_bits = _globals.AVAILABLE_BITS # we now can do hardware-aware quantization with 8:tc and 8:tc-li
    D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
    # max_device_mem can be used to check whether OOM or not
    use_profiler_prediction = args.use_profiler_prediction # use regression model to predict or load predictor
    # target model configuration
    device_info = get_device_info(device_names, device_numbers)
    comm_cost_model_dir = f'{args.comm_cost_model_dir}/{device_info}'
    cost_model_store_path = None # initialize the cost model

    if args.init_pack is None:
        model_mem_estimator, comm_cost_model, lat_cost_model, T = init_parameters_and_cost_models(config, device_names, device_numbers, cost_model_store_path, \
                                                                                                     global_bz, micro_bz, s, n, \
                                                                                                  comm_cost_model_folder=comm_cost_model_dir)
    
        args.init_pack = (model_mem_estimator, comm_cost_model, lat_cost_model, T)
        lat_cost_model.update_profiled_result(args.lat_profile_dir)
        lat_cost_model.update_profiled_prepost_result(args.lat_prepost_profile_dir)
        if args.fit:
            lat_cost_model.fit_regression_cost_model()
        else:
            if not args.use_profiler_prediction:
                lat_cost_model.load_regression_cost_model()

    lat_profile_result_path = args.lat_profile_dir
    return enumerate_best_result(args)


def llm_pq_ef_main():
    ilp_env()
    args = common_argparser()
    best_plan = main(args)
    print(best_plan)

# if __name__ == '__main__':
#     from .utils import get_final_strat_file_name
#     from .entry import convert_to_llm_pq_result2partitions
#     ilp_env()
#     args = common_argparser()
#     best_plan = main(args)
#     # model size
#     sols = {
#         'llm_pq': convert_to_llm_pq_result2partitions(best_plan)
#     }
#     model_name = args.model_name
#     folder = args.store_folder
#     file_name = get_final_strat_file_name(model_name, model_size, device_info)
#     save_with_pickle(sols, file_name, folder)
#     logger.info(f'All plans saved to {file_name} in {folder}')

