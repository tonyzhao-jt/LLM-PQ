# first use adabits, then use heuristic method to balance the load
from .. import _globals
from ..partitioner.indicator import (
    assign_omega_uniform, assign_omega_constant
)
from ..partitioner.utils import (
    assign_uniform_bit, 
    estimate_min_max_mem,
    interpret_ilp_result_i_j_b
)

from ..cost_model import (
    estimate_single_layer_mem,
    estimate_all_layer_mem,
    get_mem_with_layer_bit_pair
)

from ..utils import (
    save_with_pickle, get_default_decode_bz, partition_a_into_b_bins,
    to_weight_int8_if_tc_not_available, get_available_bits_pair, has_tc,
    get_factors
)
from ..partitioner import gen_config
from ..partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    force_zero,
    decouple_result_group,
    lat_prediction,
    get_latency_with_layer_device_bit_pair
)
from .utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE,
    create_ilp_solver,
    get_comm_payload_size, get_comm_cost,
    get_combinations
)

import copy

from .mem_utils import get_device_topo_available_mem_with_order, get_M_with_bitwidth_pair
from .mem_utils import check_memory_budget, estimate_single_device_mem
from .lat_utils import stage_pure_exe_latency
import os 
# default libs
import pickle
import numpy as np 
import math 
# setup ilp configs
import pulp
unit = _globals.MEM_UNIT
import itertools

from .MinMaxHeap import MinMaxHeap
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


def solve_ilp_pulp(L, N, BITs, M, M_d, omega):
    prob = pulp.LpProblem("max Latency Minimization Problem", pulp.LpMinimize)
    # Create a new PuLP model
    # use it to ensure the result to be contiguous


    B = len(BITs)
    z = pulp.LpVariable.dicts("z", [(i, j, b) for i in range(L) for j in range(N) for b in range(B)], cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i, b) for i in range(L) for b in range(B)], cat=pulp.LpBinary)
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(L) for j in range(N)], cat=pulp.LpBinary)

    # Define the objective function
    prob += pulp.lpSum([omega[i][b] * y[(i, b)] for i in range(L) for b in range(B)])

    # Define the constraints
    for i in range(L):
        for b in range(B):
            prob += pulp.lpSum([z[(i, j, b)] for j in range(N)]) == y[(i, b)]
    for i in range(L):
        for j in range(N):
            prob += pulp.lpSum([z[(i, j, b)] for b in range(B)]) == x[(i, j)]
    
    for i in range(L):
        prob += pulp.lpSum([x[(i, j)] for j in range(N)]) == 1
    for i in range(L):
        prob += pulp.lpSum([y[(i, b)] for b in range(B)]) == 1
    for i in range(L):
        prob += pulp.lpSum([z[(i, j, b)] for j in range(N) for b in range(B)]) == 1
    for j in range(N):
        prob += pulp.lpSum([z[(i, j, b)] * M[i][b] for i in range(L) for b in range(B)]) <= M_d[j]
    
    
    prob += x[(0,0)] == 1 # the first layer must lie in the first device
    prob += x[(L-1,N-1)] == 1
    if N > 1:
        for i in range(1, L):
            for j in range(N-1):
                for k in range(j+1, N):
                    prob += x[(i,j)] + x[(i-1,k)] <= 1
            
    # Solve the problem
    solver = create_ilp_solver(verbose_ilp, ilp_time_limit, None)
    status = prob.solve(solver)
    # prob.solve(pulp.GUROBI())
    # prob.solve()
    # Print the solution status
    if status == pulp.LpStatusOptimal:
        print("Adabits result found")
        # Print the optimal objective value
        # store the optimal solution
        result = {}
        mem_all = 0
        # print z variable result
        for i in range(L):
            for j in range(N):
                for b in range(B):
                    if z[(i, j, b)].varValue > 0:
                        # print("z[{}, {}, {}] = {}".format(i, j, b, z[(i, j, b)].varValue))
                        result[i] = (j, b)
                        # print(M[i][b])
                        mem_all += M[i][b]
        # print the memory suage of each device
        for j in range(N):
            mem_j = 0
            for i in range(L):
                for b in range(B):
                    mem_j += z[(i, j, b)].varValue * M[i][b]
            print("mem_j = {}".format(mem_j))
        
        # for i in range(L):
        #     print()
        #     for j in range(N):
        #         print(x[(i,j)].varValue, end='|')
            
        return result, pulp.value(prob.objective) 
    else:
        print("Not Feasible for adabits")
        return NOT_AVAILABLE, 1e10




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
    # omega = assign_omega_constant(group_L, BITs)
    omega = assign_omega_uniform(group_L, BITs)
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
    comm_prefill_payload, comm_decode_payload = get_comm_payload_size(lat_cost_model, s, prefill_bz, bz_decode_max, comm_multiplier)
    comm_prefill = get_comm_cost(current_D, comm_cost_model, comm_prefill_payload)
    comm_decode = get_comm_cost(current_D, comm_cost_model, comm_decode_payload) 

    # print('----- communication cost ---- ')
    # print(comm_prefill, comm_decode)
    return group_L, N, BITs, M_d, M, (l_prefill, l_decode), omega, (comm_prefill, comm_decode)


# objective
# object 1

from ..partitioner.helper import (
    lat_prediction,
)
def get_device_e2e_lat(device_rank, partition_result, bit_assignment, \
                        current_D, lat_cost_model, params, \
                        record=False, device_info_dict=None):
    D_name = current_D[device_rank]
    layer_range = partition_result[device_rank]
    bits = [(bit_assignment[2 * layer_idx], bit_assignment[2 * layer_idx + 1]) for layer_idx in range(layer_range[0], layer_range[1])]
    prefill_bz, bz_decode_max, s, mu_n = params
    
    for layer_idx in range(layer_range[0], layer_range[1]):
        atten_bit, ffn_bit = bit_assignment[2 * layer_idx], bit_assignment[2 * layer_idx+1]
        bit = atten_bit
        if record:
            device_info_dict[device_rank][atten_bit]['layers'].append(layer_idx)
            device_info_dict[device_rank][atten_bit]['sum'] += 1
    prefill_lat = stage_pure_exe_latency(D_name, bits, lat_cost_model, prefill_bz, s, 0, use_profiler_prediction=use_profiler_prediction)
    decode_lat = stage_pure_exe_latency(D_name, bits, lat_cost_model, bz_decode_max, 1, s + int(mu_n / 2), use_profiler_prediction=use_profiler_prediction)
    return prefill_lat, decode_lat

def check_latency(partition_result:dict, bit_assignment: dict, \
                    current_D, lat_cost_model, params, \
                    prefill_micro_bs_num, decode_micro_bs_num, verbose=False):
    stage_prefill_lat = []
    stage_decode_lat = []
    # negelect the intra-node communication
    prefill_bz, bz_decode_max, s, mu_n = params
    for rank, layer_range in partition_result.items():
        prefill_lat, decode_lat = get_device_e2e_lat(rank, partition_result, bit_assignment, current_D, lat_cost_model, params)
        stage_prefill_lat.append(prefill_lat)
        stage_decode_lat.append(decode_lat)
        if prefill_lat < 0 or decode_lat < 0:
            return None 
    # sum
    prefill_lat = sum(stage_prefill_lat)
    decode_lat = sum(stage_decode_lat)
    # maximum latency
    max_prefill_lat = max(stage_prefill_lat)
    max_decode_lat = max(stage_decode_lat)
    # objective
    prefill_time = prefill_lat + max_prefill_lat * (prefill_micro_bs_num - 1)
    decode_time = decode_lat + max_decode_lat * (decode_micro_bs_num - 1) * (mu_n - 1)
    # latency equals
    e2e_lat = prefill_time + decode_time
    if verbose:
        print(stage_decode_lat, stage_prefill_lat, e2e_lat)
    return e2e_lat

# object 2
def check_bitwidth_omega(bit_assignment: dict, available_bits: list, omega):
    bit_assignment_list = list(bit_assignment.values())
    new_sum = 0
    T = len(bit_assignment_list) // 2
    for layer_idx in range(T):
        bit = bit_assignment_list[2 * layer_idx]
        omega_val = omega[layer_idx, available_bits.index(bit)]
        new_sum += omega_val
    return new_sum


def change_partition_result(partition_result:dict, pioneer_rank, straggler_rank, num_convert):
    copied_partition_result = copy.deepcopy(partition_result)
    # each rank value
    for rank, value in partition_result.items():
        if rank == pioneer_rank:
            copied_partition_result[rank] = (value[0], value[1] + num_convert) # increase layer number
        elif rank == straggler_rank:
            copied_partition_result[rank] = (value[0] + num_convert, value[1]) # reduce layer number

    # get new partition_result
    idx = 0
    new_partition_result = {}
    for rank, value in copied_partition_result.items():
        start_rank = idx 
        end_rank = idx + value[1] - value[0]
        new_partition_result[rank] = (start_rank, end_rank)
        idx += value[1] - value[0]
    return new_partition_result


def exchange_precisions(original_bit_assignment:dict, pioneer_layer_idx:int, straggler_idxs:list, conversion:list):
    original_bit_assignment = copy.deepcopy(original_bit_assignment) 
    num_convert = len(straggler_idxs)
    layer_num = len(original_bit_assignment) // 2
    original_bitwidths_values = list(original_bit_assignment.values())
    original_bitwidths = [original_bitwidths_values[idx * 2] for idx in range(layer_num)]
    # minmax
    min_straggler_layer_idx = min(straggler_idxs)
    max_straggler_layer_idx = max(straggler_idxs)
    # get precisions
    pi_precs, straggler_precs = conversion
    candidate_bit_assignment = []
    for insert_idx in range(num_convert):
        copied_bit_assignment = copy.deepcopy(original_bitwidths)
        if pioneer_layer_idx < min_straggler_layer_idx: # pioneer layer is before the straggler
            copied_bit_assignment.pop(pioneer_layer_idx)
            for _ in range(num_convert):
                copied_bit_assignment.insert(pioneer_layer_idx, straggler_precs)
            # pop all straggler precisions
            straggler_idxs_len = len(straggler_idxs)
            ref_idx = 0
            pop_idx = 0
            while ref_idx < straggler_idxs_len:
                strag_layer_idx = straggler_idxs[ref_idx]
                try:
                    try_insert_idx = strag_layer_idx + (num_convert - 1) + pop_idx # new idx
                    copied_bit_assignment.pop(try_insert_idx) # offset
                    if ref_idx == insert_idx:
                        copied_bit_assignment.insert(try_insert_idx, pi_precs)
                        # remove on add one equals not add
                    else:
                        # no insert idx, 
                        # as asecnding, the suceed -=1
                        pop_idx -= 1
                except:
                    import pdb; pdb.set_trace()
                ref_idx += 1
        elif pioneer_layer_idx > max_straggler_layer_idx: # pioneer layer is after the straggler
            copied_bit_assignment.pop(pioneer_layer_idx)
            for _ in range(num_convert):
                copied_bit_assignment.insert(pioneer_layer_idx - num_convert, straggler_precs)
            # pop all straggler precisions
            pop_idx = 0 # when ever an element is poped, the index should be decreased
            for ref_idx, strag_layer_idx in enumerate(straggler_idxs):
                copied_bit_assignment.pop(strag_layer_idx)
                if ref_idx == insert_idx:
                    copied_bit_assignment.insert(strag_layer_idx + pop_idx, pi_precs)
                else:
                    pop_idx -= 1
        else:
            print(pioneer_layer_idx, straggler_idxs)
            raise ValueError('pioneer layer is in the middle of straggler layers')

        candidate_bit_assignment.append(copied_bit_assignment)

    # convert candidate bit assignment to dict
    candidate_bit_assignment_dict_list = []
    for candidate_bit_assignment_item in candidate_bit_assignment:
        candidate_bit_assignment_dict = {}
        for layer_idx, bit in enumerate(candidate_bit_assignment_item):
            candidate_bit_assignment_dict[layer_idx * 2] = bit
            candidate_bit_assignment_dict[layer_idx * 2 + 1] = bit
        candidate_bit_assignment_dict_list.append(candidate_bit_assignment_dict)
    return candidate_bit_assignment_dict_list

def are_dicts_equal(dict1, dict2):
    """
    Returns True if the two dictionaries have the same key-value pairs, False otherwise.
    """
    if len(dict1) != len(dict2):
        return False
    for key, value in dict1.items():
        if key not in dict2 or dict2[key] != value:
            return False
    return True

def check_new_plan_valid(old_partition, old_bit_ass, 
                         new_partition, new_bit_ass):
    # check if the partition has the same layer numbers
    old_layer_sum = 0
    new_layer_sum = 0
    for rank, layer_range in old_partition.items():
        old_layer_sum += layer_range[1] - layer_range[0]
        new_layer_sum += new_partition[rank][1] - new_partition[rank][0]
    # check bitwidth
    assert old_layer_sum == new_layer_sum, "layer num sum should same"
    assert len(old_bit_ass) == len(new_bit_ass), "bit assign should same"



def calculate_stage_latency(current_D, partition_result, bit_assignment, \
                            lat_cost_model, params, \
                            comm_prefill, comm_decode, \
                            prefill_micro_bs_num, decode_micro_bs_num, \
                            device_info_dict=None, record=True, verbose=False):
    prefill_bz, bz_decode_max, s, mu_n = params
    lat_stages = []
    # get latency of all stages
    stage_time_dict = {}
    stage_time_dict_decomposed = {}
    for device_rank, device_name in current_D.items():
        prefill_stage_lat, decode_stage_lat = get_device_e2e_lat(device_rank, partition_result, bit_assignment, \
                    current_D, lat_cost_model, params, record=record, device_info_dict=device_info_dict)
        
        # communication time
        prefill_comm = comm_prefill[device_rank]
        decode_comm = comm_decode[device_rank]
        # max to get the time
        prefill_stage_lat = max(prefill_stage_lat, prefill_comm)
        decode_stage_lat = max(decode_stage_lat, decode_comm)

        # prefill_time[device_rank] = prefill_stage_lat
        # decode_time[device_rank] = decode_stage_lat

        # heap_prefill.insert((prefill_stage_lat, device_rank))
        # heap_decode.insert((decode_stage_lat, device_rank))
        # heap_weighted.insert((prefill_stage_lat * (prefill_micro_bs_num - 1) + (mu_n - 1) * (decode_micro_bs_num - 1) * decode_stage_lat, device_rank))
        stage_time = prefill_stage_lat * (prefill_micro_bs_num - 1) + (mu_n - 1) * (decode_micro_bs_num - 1) * decode_stage_lat
        stage_time_dict[device_rank] = stage_time
        stage_time_dict_decomposed[device_rank] = (prefill_stage_lat, decode_stage_lat)
        lat_stages.append((stage_time, device_rank))
    if verbose:
        print("better plan", stage_time_dict)
    return lat_stages, (stage_time_dict, stage_time_dict_decomposed)


# possible conversion pairs
convert_pairs = [
    (16, ['8:tc-li', 2]),
    (16, [4, 4]),
    ('8:tc-li', [4, 2]),
]

def shaq_h_internal_main(num_hidden_layers, cost_model_pack, bz_pack, current_D):
    (global_bz, prefill_bz, bz_decode_max) = bz_pack 
    group_L, N, BITs, M_d, M, (l_prefill, l_decode), omega, (comm_prefill, comm_decode) \
          = prepare_for_ilp(num_hidden_layers, current_D, available_bits, cost_model_pack, bz_pack)
    plan, obj_value = solve_ilp_pulp(group_L, N, BITs, M, M_d, omega)
    plan = decouple_result_group(group_size, plan)
    res = {
        'plan': plan,
        'obj': obj_value
    }
    prefill_micro_bs_num = math.ceil(global_bz / prefill_bz)
    decode_micro_bs_num = math.ceil(global_bz / bz_decode_max)

    if plan != NOT_AVAILABLE:
        res['plan'] = interpret_ilp_result_i_j_b(res['plan'], BITs)
        # do heuristic to balance the load
        from ..partitioner.helper import (
            lat_prediction,
        )
        # get latency of all stages
        partition_result = res['plan']['partition_result']
        bit_assignment = res['plan']['bit_assignment']
        res['D'] = current_D
        res['prefill_bz'] = prefill_bz
        res['bz_decode_max'] = bz_decode_max
        model_mem_estimator, comm_cost_model, lat_cost_model = cost_model_pack
        check_memory_budget(res, model_mem_estimator, name='shaq_h')
        device_num = len(current_D)

        # create a heap to store the latency of each stage
        # heap_weighted = MinMaxHeap(device_num)
        
        '''
            Collect
            rank: {
                16: {layers: [layer_idx, layer_idx, ...], sum: n},
                8: {layers: [layer_idx, layer_idx, ...], sum: n},
            }
        '''
        device_info_dict = {device_rank: {bit: {
            'layers': [],
            'sum': 0,
            # 'heap': MinMaxHeap(200) # big enough
        } for bit in available_bits} for device_rank in current_D}
        params = (prefill_bz, bz_decode_max, s, mu_n)

        lat_stages, (stage_time_dict, stage_time_dict_decomposed) = calculate_stage_latency(current_D, partition_result, bit_assignment, \
                            lat_cost_model, params, \
                            comm_prefill, comm_decode, \
                            prefill_micro_bs_num, decode_micro_bs_num, \
                            device_info_dict=device_info_dict, record=True)
        
        step = 1 # each time change 1 precisions
        obj1 = check_latency(partition_result, bit_assignment, \
                            current_D, lat_cost_model, params, \
                            prefill_micro_bs_num, decode_micro_bs_num)
        if obj1 is None:
            res = {
                'obj': float("inf"),
            }
            return res # the case when prediction negative value, when bs = 1, could happend
        obj2 = check_bitwidth_omega(bit_assignment, available_bits, omega)
        old_obj = obj1 + theta * obj2
        
        # we cannot start from pipedge as when loose the memory bound, it will place all layers to the same device(powerful device)
        # N is the device number, L is the number of layers
        # Complexity: O(N/2 * L/2) * O(2 * 1 + C * subL^{nums + 1} + 2 * NlogN) # update then
        # worst case: O(N/2 * L/2)
        # as always higher substitue lower, worst case it evenly has N/2 stragglers and pioneers, and each layer in pioneer has subL layers for conversion
        
        optimal_obj = old_obj
        while True:
            # each time got the straggler and pioneer
            #O(NlogN)
            sorted_lat_stages = sorted(lat_stages, key=lambda x: x[0])
            straggler = sorted_lat_stages[-1]
            # candidate pioneers
            candidate_pioneers = sorted_lat_stages[:-1]
            # O(N)
            # check whether there is a pioneer that can help
            better_candidate = None
            for pioneer in candidate_pioneers:
                straggler_rank = straggler[1] # always stragler, if you cannot speed up straggler, you can never speed up
                straggler_lat = straggler[0]
                pioneer_rank = pioneer[1]
                pioneer_lat = pioneer[0]
                original_max_lat = max(straggler_lat, pioneer_lat)
                # get precision numbers, and check from high to low
                # iterate among all possible conversion pairs O(C) ~ O(1)
                for pioneer_prec, (strag_prec, num_required) in convert_pairs:
                    # first check whether this pair is possible, if not, continue
                    if device_info_dict[pioneer_rank][pioneer_prec]['sum'] <= 0:
                        continue
                    if device_info_dict[straggler_rank][strag_prec]['sum'] <= num_required:
                        continue
                    # available, get pairs
                    pioneer_prec_layers = device_info_dict[pioneer_rank][pioneer_prec]['layers']
                    straggler_prec_layers = device_info_dict[straggler_rank][strag_prec]['layers']
                    possible_straggler_layer_pairs = get_combinations(straggler_prec_layers, num_required)
                    
                    conversion = (pioneer_prec, strag_prec)
                    new_partition_result = change_partition_result(partition_result, pioneer_rank, straggler_rank, num_required)

                    # worst case O(subL)
                    # find the one with smallest omega value
                    obj1_new = None 
                    obj2_optimal = 1e10
                    layer_change_candidate = None
                    for pioneer_layer_idx in pioneer_prec_layers:
                        # worst case O(subL!/((subL-nums)! * nums!)) << O(subL^{nums})
                        for straggler_layer_idxs in possible_straggler_layer_pairs:
                            # all possible new placesment O(nums)
                            # print(straggler, pioneer)
                            candidates = exchange_precisions(bit_assignment, pioneer_layer_idx, straggler_layer_idxs, conversion)
                            # get best one that minimize the objective
                            for candidate in candidates:
                                # print(are_dicts_equal(bit_assignment, candidate))
                                # same moving resuling in the same speed. no need to repeat
                                if obj1_new is None:
                                    try:
                                        obj1_new = check_latency(new_partition_result, candidate, \
                                                            current_D, lat_cost_model, params, \
                                                            prefill_micro_bs_num, decode_micro_bs_num)
                                        # print("Move {} layers from {} to {}".format(num_required, straggler_rank, pioneer_rank))
                                        # print("Latency new", obj1_new)
                                    except:
                                        import pdb; pdb.set_trace()
                                obj2_new = check_bitwidth_omega(candidate, available_bits, omega) * theta
                                if obj2_new < obj2_optimal:
                                    # print("new obj2:", obj2_new)
                                    obj2_optimal = obj2_new
                                    better_bit_assignment = candidate
                                    better_layer_partition = new_partition_result
                                    layer_change_candidate = (pioneer_layer_idx, straggler_layer_idxs)
                    
                    new_obj = obj1_new + obj2_optimal if obj1_new is not None else None
                    if new_obj is not None and new_obj < optimal_obj:
                        # check this memory is valid or not
                        straggler_layers_range = better_layer_partition[straggler_rank]
                        pioneer_layers_range = better_layer_partition[pioneer_rank]
                        strag_new_mem = estimate_single_device_mem(straggler_layers_range, better_bit_assignment, model_mem_estimator)
                        pioneer_new_mem = estimate_single_device_mem(pioneer_layers_range, better_bit_assignment, model_mem_estimator)
                        strag_D_mem = M_d[straggler_rank]
                        pioneer_D_mem = M_d[pioneer_rank]
                        # check whether convertion is valid
                        if strag_new_mem > strag_D_mem or pioneer_new_mem > pioneer_D_mem:
                            # print("Invalid conversion, exceed memory")
                            continue
                        optimal_obj = new_obj
                        # update the best one
                        better_candidate = {
                            'obj': new_obj,
                            'layer_changed': layer_change_candidate,
                            'involved_devices': [straggler_rank, pioneer_rank],
                            'new_sol': {
                                'partition_result': better_layer_partition,
                                'bit_assignment': better_bit_assignment
                            }
                        }

                                
                # enumrated all possible conversion and layers already
                # has candiate for this conversion pair
                # in an ascending order, thus first one find then break
                if better_candidate is not None:
                    check_new_plan_valid(partition_result, bit_assignment, \
                        better_candidate['new_sol']['partition_result'], better_candidate['new_sol']['bit_assignment'])
                    # check the result
                    print("Update")
                    layer_change_candidate = better_candidate['layer_changed']
                    straggler_rank, pioneer_rank = better_candidate['involved_devices']
                    new_partition_result = better_candidate['new_sol']['partition_result']
                    new_bit_assignment = better_candidate['new_sol']['bit_assignment']
                    new_obj = better_candidate['obj']
                    # update
                    bit_assignment = new_bit_assignment
                    partition_result = new_partition_result
                    # update obj
                    # print("Layer changed from {} to {}".format(straggler_rank, pioneer_rank))
                    # print("Layer changed:", layer_change_candidate)
                    print(f'obj update from {optimal_obj} to {new_obj}')
                    optimal_obj = new_obj
                    break 
                else:
                    continue

            if better_candidate is not None: # means the obj is updated
                # continue to update the result
                # update
                # renew the latency
                device_info_dict = {device_rank: {bit: {
                    'layers': [],
                    'sum': 0,
                    # 'heap': MinMaxHeap(200) # big enough
                } for bit in available_bits} for device_rank in current_D}
                lat_stages, (stage_time_dict, stage_time_dict_decomposed) = calculate_stage_latency(current_D, partition_result, bit_assignment, \
                            lat_cost_model, params, \
                            comm_prefill, comm_decode, \
                            prefill_micro_bs_num, decode_micro_bs_num, \
                            device_info_dict=device_info_dict, record=True, verbose=True)
                # print("lat stages", lat_stages)
                better_candidate = None # reset 
                continue
            else:
                # no available find, done.
                break
                    
    
    res['plan']['partition_result'] = partition_result
    res['plan']['bit_assignment'] = bit_assignment
    res['obj'] = optimal_obj
    return res
    # device_info = get_device_info(device_names, device_numbers)
    # file_name = f'adaptive_bit_' + model_size + '_' + device_info + '.pkl'
    # folder = '/workspace/qpipe/scripts/strategy'
    # save_with_pickle(best_plan, file_name, folder)


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
    if args.force_fixed_D:
        # dont change device order
        permutations = [device_name_with_its_number]
    elif args.force_reverse_D:
        permutations = [device_name_with_its_number[::-1]]
    num_hidden_layers = len(T) // 2
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
            res = shaq_h_internal_main(num_hidden_layers, cost_model_pack, bz_pack, current_D)
            # print(res['obj'], best_plan['obj'])
            if res['obj'] < best_plan['obj']:
                print("Better Plan Generated", res['obj'])
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
                    'name': 'shaq'
                }
                check_memory_budget(best_plan, model_mem_estimator, name='shaq_h')
            comm_cost_model.clear_device_rank_map()
    # log the best plan
    print('best plan: ', best_plan)
    # save the result
    # file_name = f'shaq_' + model_size + '_' + device_info + '.pkl'
    # folder = '/workspace/qpipe/scripts/strategy'
    # save_with_pickle(best_plan, file_name, folder)
    return best_plan

'''
    Initiailization
'''
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
omega_file = None
group_size = 1
ilp_seed = 0
ilp_time_limit = 20
verbose_ilp = False
comm_multiplier = 1
ilp_tolerance = None
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
    global omega_file
    global ilp_seed
    global group_size
    global ilp_time_limit
    global verbose_ilp
    global comm_multiplier
    # global variables

    omega_file = args.omega_file
    ilp_seed = args.ilp_seed
    group_size = args.adapp_group_size
    ilp_tolerance = args.ilp_tolerance
    ilp_time_limit = args.ilp_time_limit
    verbose_ilp = args.debug
    if ilp_tolerance is not None:
        pulp.LpSolverDefault.eps = ilp_tolerance
    comm_multiplier = args.comm_multiplier
    
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
    available_bits = _globals.AVAILABLE_BITS_WO_INFO # we now can do hardware-aware quantization with 8:tc and 8:tc-li
    if args.adabits_tc:
        print("Use AdaBits-TC")
        available_bits = _globals.AVAILABLE_BITS
    D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
    # max_device_mem can be used to check whether OOM or not
    use_profiler_prediction = args.use_profiler_prediction # use regression model to predict or load predictor
    # target model configuration
    device_info = get_device_info(device_names, device_numbers)
    comm_cost_model_dir = f'{args.comm_cost_model_dir}/{device_info}'
    cost_model_store_path = None # initialize the cost model

    if args.init_pack:
        model_mem_estimator, comm_cost_model, lat_cost_model, T = args.init_pack 
    if args.debug:
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
    cost_model_pack = (model_mem_estimator, comm_cost_model, lat_cost_model)
    # common parts above
    # assign bits
    bit_map = {}
    assign_uniform_bit(T, 16, bit_map)
    initial_mem = estimate_all_layer_mem(model_mem_estimator, T, bit_map)
    max_model_mem, min_model_mem = estimate_min_max_mem(model_mem_estimator, T)

    num_hidden_layers = len(T) // 2
    num_devices = len(D)

    # set available bits
    num_device_all = num_devices
    bz_decode_max = get_default_decode_bz(global_bz, num_device_all)
    return enumerate_best_result(args)


def shaq_h_main():
    ilp_env()
    args = common_argparser()
    args.debug = True
    best_plan = main(args)
    print(best_plan)
