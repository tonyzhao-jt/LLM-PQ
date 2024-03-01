
from .. import _globals
from ..partitioner.indicator import (
    assign_omega_uniform,
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
    to_weight_int8_if_tc_not_available, get_available_bits_pair, has_tc
)
from ..partitioner import gen_config
from ..partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    force_zero,
    decouple_result_group,
    lat_prediction
)
from .utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE,
    create_ilp_solver
)
from .mem_utils import get_device_topo_available_mem_with_order, get_M_with_bitwidth_pair
import os 
# default libs
import pickle
import numpy as np 

# setup ilp configs
import pulp
unit = _globals.MEM_UNIT

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




def prepare_for_ilp(num_hidden_layers, D, available_bits, bz_pack, model_mem_estimator):
    time_mult_times = _globals.TIME_MULT_TIMES
    global_bz, prefill_bz, bz_decode_max = bz_pack
    L = num_hidden_layers # in partition, regard as a whole
    # group_L
    assert L % group_size == 0, f'L should be divisible by group_size, but L={L}, group_size={group_size}'
    group_L = L // group_size
    N = len(D) # number of devices
    BITs = get_available_bits_pair(available_bits)
    
    # get memory requirement of each layer
    M = get_M_with_bitwidth_pair(BITs, model_mem_estimator, group_L, group_size)
    M_d = get_device_topo_available_mem_with_order(D, model_mem_estimator, prefill_bz, bz_decode_max, time_mult_times=time_mult_times)

    # omega
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
        try:
            if omega_loaded.shape != omega.shape:
                print(omega_loaded.shape, omega.shape)
                raise ValueError('omega shape mismatched')
            omega = omega_loaded
        except:
            import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # for i in range(N):
    #     device_name = D[i]
    #     has_tc_bit = has_tc(device_name)
    #     if not has_tc_bit:
    #         # reset the available bits
    #         new_BITs = get_available_bits_pair(_globals.AVAILABLE_BITS_WO_INFO)
    #         # get the index of new_bits in the original BITs
    #         new_BITs_idx = [BITs.index(bit_pair) for bit_pair in new_BITs]
    #         excepted_indexes = list(set(range(len(BITs))) - set(new_BITs_idx))
    #         # change the omega outside the new_bits_idx to be extremely large
    #         omega[i][excepted_indexes] = 1e10
    return group_L, N, BITs, M_d, M, omega

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
        lat_cost_model.update_profiled_result(args.lat_profile_dir)
        if not use_profiler_prediction:
            lat_cost_model.load_regression_cost_model()

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
    plan = decouple_result_group(group_size, plan)
    res = {
        'plan': plan,
        'obj': obj_value
    }
    if plan != NOT_AVAILABLE:
        res['plan'] = interpret_ilp_result_i_j_b(res['plan'], BITs)
    best_plan = {
        'prefill_bz': prefill_bz,
        'bz_decode_max': bz_decode_max,
        'bz_decode_bss': strat,
        'device_names': device_names,
        'device_numbers': device_numbers,
        'plan': res['plan'],
        'obj': res['obj'],
        'D': D,
        'maps': None,
        'name': 'adabits'
    }
    # device_info = get_device_info(device_names, device_numbers)
    # file_name = f'adaptive_bit_' + model_size + '_' + device_info + '.pkl'
    # folder = '/workspace/qpipe/scripts/strategy'
    # save_with_pickle(best_plan, file_name, folder)
    return best_plan

# if __name__ == '__main__':
#     ilp_env()
#     args = common_argparser()
#     args.debug = True
#     best_plan = main(args)
#     print(best_plan)
