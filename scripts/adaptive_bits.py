from qllm.models.OPT import OPTForCausalLMSeq
from qllm.models import opt

from qllm.models.BLOOM import BloomForCausalLMSeq
from qllm.models import bloom

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
    estimate_all_layer_mem,
    get_mem_with_layer_bit_pair
)

from qpipe.utils import (
    save_with_pickle, get_default_decode_bz, partition_a_into_b_bins,
    to_weight_int8_if_tc_not_available, get_available_bits_pair, has_tc
)

from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    force_zero
)
from utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE
)

# default libs
import pickle
import os 
import numpy as np 

# setup ilp configs
import pulp
import gurobipy as gp
ilp_env()
args = common_argparser()

unit = qpipe._globals.MEM_UNIT
time_mult_times = qpipe._globals.TIME_MULT_TIMES

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
    status = prob.solve(pulp.GUROBI(MIPGap=0.01))
    # prob.solve(pulp.GUROBI())
    # prob.solve()
    # Print the solution status
    if status == pulp.LpStatusOptimal:
        # Print the optimal objective value
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


def prepare_for_ilp(num_hidden_layers, D, available_bits, bz_pack, model_mem_estimator):
    global_bz, prefill_bz, bz_decode_max = bz_pack
    L = num_hidden_layers # in partition, regard as a whole
    N = len(D) # number of devices
    BITs = get_available_bits_pair(available_bits)
    M_d = np.array([get_single_device_mem_constraints(device_name) for d_rank, device_name in D.items()]) 
    mem_bits_vector = get_mem_with_layer_bit_pair(BITs, model_mem_estimator)
    M = np.tile(mem_bits_vector, (L, 1))

    # reduce the embedding size on device 0
    post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size_with_bz(prefill_bz, bz_decode_max, unit='MB')[0] 
    M_d[0] -= post_pre_mem
    M_d -= time_mult_times * temp_tensor_mem # torch may not release the tensor immediately, modify the 2 to ensure won't oom
    M_d = np.floor(M_d).astype(int) # floor
    M = np.ceil(M).astype(int) # ceil
    # omega
    omega = assign_omega_uniform(L, BITs)
    if omega_file is not None:
        # open and load with pickle
        with open(omega_file, 'rb') as f:
            omega_loaded = pickle.load(f)
        # check whether the shape is matched, as raise error
        if omega_loaded.shape != omega.shape:
            raise ValueError('omega shape mismatched')
        omega = omega_loaded

    # for i in range(N):
    #     device_name = D[i]
    #     has_tc_bit = has_tc(device_name)
    #     if not has_tc_bit:
    #         # reset the available bits
    #         new_BITs = get_available_bits_pair(qpipe._globals.AVAILABLE_BITS_WO_INFO)
    #         # get the index of new_bits in the original BITs
    #         new_BITs_idx = [BITs.index(bit_pair) for bit_pair in new_BITs]
    #         excepted_indexes = list(set(range(len(BITs))) - set(new_BITs_idx))
    #         # change the omega outside the new_bits_idx to be extremely large
    #         omega[i][excepted_indexes] = 1e10

    return L, N, BITs, M_d, M, omega

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
omega_file = None
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
    # global variables

    
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
    available_bits = qpipe._globals.AVAILABLE_BITS_WO_INFO # we now can do hardware-aware quantization with 8:tc and 8:tc-li
    D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
    # max_device_mem can be used to check whether OOM or not
    use_profiler_prediction = args.use_profiler_prediction # use regression model to predict or load predictor
    # target model configuration
    device_info = get_device_info(device_names, device_numbers)
    comm_cost_model_dir = f'{args.comm_cost_model_dir}/{device_info}'
    cost_model_store_path = None # initialize the cost model

    model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names, cost_model_store_path, \
                                                                                                     global_bz, micro_bz, s, n, \
                                                                                                    comm_cost_model_folder=comm_cost_model_dir)
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
    strat = partition_a_into_b_bins(global_bz, num_device_all)
    prefill_bz = bz_decode_max


    # print(initial_mem, max_device_mem)
    # start quantization
    # assign indicator
    # ILP to get the optimal bit map
    # verysimple, choose bits, minimize the sum of indicator, satsify the constraints of the memory
    # Given L layers: 0,..L-1
    # each layer can have bits b from a set of bits B: 2,4,8,16
    # each layer has a quantization sensitivity indicator i_(l,b) for different quantization bits b
    # the total memory requirement of the model is the sum of the memory requirement of each layer M(l,b)
    # try to minimize the sum of indicator i_(l,b) while satisfying the memory constraint
    bz_pack = (global_bz, prefill_bz, bz_decode_max)
    L, N, BITs, M_d, M, omega = prepare_for_ilp(num_hidden_layers, D, available_bits, bz_pack, model_mem_estimator)
    plan, obj_value = solve_ilp_pulp(L, N, BITs, M, M_d, omega)
    res = {
        'plan': plan,
        'obj': obj_value
    }
    if plan != NOT_AVAILABLE:
        res['plan'] = interpret_ilp_result_i_j_b(res['plan'], BITs)
    best_plan = {
        'prefill_bz': bz_decode_max,
        'bz_decode_max': bz_decode_max,
        'bz_decode_bss': strat,
        'device_names': device_names,
        'device_numbers': device_numbers,
        'plan': res['plan'],
        'obj': res['obj'],
        'D': D,
        'maps': None
    }
    # device_info = get_device_info(device_names, device_numbers)
    # file_name = f'adaptive_bit_' + model_size + '_' + device_info + '.pkl'
    # folder = '/workspace/qpipe/scripts/strategy'
    # save_with_pickle(best_plan, file_name, folder)
    return best_plan

if __name__ == '__main__':
    ilp_env()
    args = common_argparser()
    best_plan = main(args)
    print(best_plan)
