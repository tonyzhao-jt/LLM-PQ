# qllm libs
# qpipe libs
from .. import _globals
from ..partitioner.utils import (
    interpret_ilp_result_i_j_b
)
from ..cost_model import (
    get_mem_with_layer_bit_pair
)
from ..partitioner.helper import (
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    decouple_result_group,
    get_latency_with_layer_device_bit_pair
)

from ..utils import (
    get_default_decode_bz,
    partition_a_into_b_bins
)

from .utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE,
    create_ilp_solver
)
from ..partitioner import gen_config

# default libs
import numpy as np 
import os 
# setup ilp configs
import pulp
import gurobipy as gp
from .mem_utils import get_device_topo_available_mem_with_order, get_M_with_bitwidth_pair

unit = _globals.MEM_UNIT
slo_rate = _globals.SLO_RATE

def solve_ilp_pulp(L, N, BITs, M, M_d, l, comm):
    prob = pulp.LpProblem("max Latency Minimization Problem", pulp.LpMinimize)
    # Create a new PuLP model
    B = len(BITs)
    z = pulp.LpVariable.dicts("z", [(i, j) for i in range(L) for j in range(N)], cat=pulp.LpBinary)
    LAT = pulp.LpVariable.dicts("LAT", [j for j in range(N)], lowBound=0, cat=pulp.LpContinuous)
    LAT_max = pulp.LpVariable("LAT_max", lowBound=0, cat=pulp.LpContinuous)

    # Define the objective function
    prob += LAT_max

    mem = M[0].item()
    # Define the constraints
    for i in range(L):
        prob += pulp.lpSum([z[(i, j)] for j in range(N)]) == 1
    for j in range(N):
        prob += pulp.lpSum([z[(i, j)] * mem for i in range(L)]) <= M_d[j]
        prob += pulp.lpSum([z[(i, j)] * l[i][j][0] for i in range(L)]) <= LAT[j]
        prob += LAT_max >= LAT[j]
        prob += LAT_max >= comm[j] 
    
    solver = create_ilp_solver(verbose_ilp, ilp_time_limit, ilp_tolerance)
    status = prob.solve(solver)

    if status == pulp.LpStatusOptimal:
        # Print the solution status
        print("Pipedge Result found")
        # store the optimal solution
        result = {}
        bit_pair = BITs[0]
        # print z variable result
        for i in range(L):
            for j in range(N):
                if z[(i, j)].varValue > 0:
                    index_ = all_available_pairs.index(bit_pair)
                    result[i] = (j, index_)
                    # print("layer {} is assigned to device {} with bit {}".format(i, j, bit_pair))
        return result, pulp.value(prob.objective)
    else:
        print("Pipedge Result not feasible")
        return NOT_AVAILABLE, 1e10

    



def get_comm(D, comm_cost_model, comm_size):
    device_length = len(D)
    comm = np.zeros(device_length)
    for idx in range(device_length):
        comm[idx] = comm_cost_model.predict_comm_time(idx, (idx + 1) % device_length, comm_size)
    return comm


def prepare_for_ilp(num_hidden_layers, D, chosen_bit, cost_model_pack, bz_pack):
    time_mult_times = _globals.TIME_MULT_TIMES
    model_mem_estimator, comm_cost_model, lat_cost_model = cost_model_pack
    global_bz, prefill_bz, bz_decode_max = bz_pack
    L = num_hidden_layers # in partition, regard as a whole
    N = len(D) # number of devices
    assert L % group_size == 0, f'L should be divisible by group_size, but L={L}, group_size={group_size}'
    group_L = L // group_size

    BITs = [(chosen_bit, chosen_bit)]
    '''
        Constraint related
    '''

    # get memory requirement of each layer
    M = get_M_with_bitwidth_pair(BITs, model_mem_estimator, group_L, group_size)
    M_d = get_device_topo_available_mem_with_order(D, model_mem_estimator, prefill_bz, bz_decode_max, time_mult_times=time_mult_times)

    # latency
    l = np.zeros((group_L, N, len(BITs)))
    lat_device_bits_matrix = get_latency_with_layer_device_bit_pair(D, BITs, lat_cost_model, bz_decode_max, 1, s + int(mu_n / 2), \
                                                                    use_profiler_prediction=use_profiler_prediction)
    for i in range(group_L):
        l[i, :, :] = lat_device_bits_matrix * group_size
    
    # comm
    comm_decode = (model_mem_estimator.h1 * bz_decode_max * 1)  * 2 / 1024 / 1024
    comm = get_comm(D, comm_cost_model, comm_decode) * comm_multiplier
    return group_L, N, BITs, M_d, M, l, comm

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
all_available_pairs = []
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
    global all_available_pairs
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

    model_mem_estimator, comm_cost_model, lat_cost_model, T = args.init_pack

    all_device_nums = sum(device_numbers)
    bz_decode = get_default_decode_bz(global_bz, all_device_nums)
    chosen_bit = args.pe_bit
    assert chosen_bit in available_bits, f'chosen bit {chosen_bit} not in available bits {available_bits}'

    num_hidden_layers = len(T) // 2
    num_device_all = len(D)
    bz_decode_max = get_default_decode_bz(global_bz, num_device_all)
    strat = partition_a_into_b_bins(global_bz, num_device_all)
    prefill_bz = bz_decode_max
    bz_pack = (global_bz, prefill_bz, bz_decode_max)
    cost_model_pack = (model_mem_estimator, comm_cost_model, lat_cost_model)
    L, N, BITs, M_d, M, l, comm = prepare_for_ilp(num_hidden_layers, D, chosen_bit, cost_model_pack, bz_pack)
    all_available_pairs = BITs
    plan, obj_value = solve_ilp_pulp(L, N, BITs, M, M_d, l, comm)

    if plan != NOT_AVAILABLE:
        plan = decouple_result_group(group_size, plan)
        plan = interpret_ilp_result_i_j_b(plan, BITs)

    res = {
        'plan': plan,
        'obj': obj_value
    }
    best_plan = {
        'prefill_bz': bz_decode,
        'bz_decode_max': bz_decode,
        'bz_decode_bss': strat,
        'device_names': device_names,
        'device_numbers': device_numbers,
        'plan': res['plan'],
        'obj': res['obj'],
        'D': D,
        'maps': None
    }
    return best_plan
    # print(result)
    # save the result
    # file_name = f'pipeedge_' + model_size + '_' + device_info + '.pkl'
    # folder = '/workspace/qpipe/scripts/strategy'
    # save_with_pickle(best_plan, file_name, folder)

