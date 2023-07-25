from .adabits import main as adaptive_bits_main
from .shaq import main as shaq_main
from .shaq_efficient import main as shaq_ef_main
from .shaq_heuristic import main as shaq_h_main
from .pipeedge_ilp import main as pipeedge_ilp_main
from .uniform import main as uniform_main
from .. import _globals 
# arg parser
from .utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE,
    get_final_strat_file_name,
    convert_to_shaq_result2partitions,

)
from ..utils import save_with_pickle, has_tc
from ..partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    get_device_info,
    lat_prediction,
)
from ..partitioner import gen_config
from ..cost_model import (
    estimate_single_layer_mem,
)
import os 
import copy 
import logging
import math 
import time 

logger = logging.getLogger(__name__)

from .lat_utils import run_simu
from .mem_utils import get_device_topo_available_mem_with_order

# for debug
def check_minimum_bit_of_sols(sol):
    bit_assignment = sol['plan']['bit_assignment']
    minimum_bit = 16
    for k, v in bit_assignment.items():
        if type(v) is str:
            v = 8
        if v < minimum_bit:
            minimum_bit = v
    print("minimum_bit: ", minimum_bit)

# first make sure the partition is within the memory budget
def check_memory_budget_single_device(device_mem, device_rank, layers_range, bit_assignment, model_mem_estimator, bs_pack):
    time_mult_times = _globals.TIME_MULT_TIMES
    i, j = layers_range
    prefill_bz, bz_decode_max = bs_pack
    # k % 2 means shard
    i_to_j_mem = sum([estimate_single_layer_mem(model_mem_estimator, 0, bit_assignment[k * 2]) for k in range(i, j)]) + \
        sum([estimate_single_layer_mem(model_mem_estimator, 1, bit_assignment[k * 2 + 1]) for k in range(i, j)])

    # print(time_mult_times * temp_tensor_mem)
    # print(i_to_j_mem, device_mem)
    if i_to_j_mem > device_mem:
        print(f"memory budget exceeded for device {device_rank}, {i_to_j_mem} > {device_mem}")
        return False
    return True

def check_memory_budget(res, model_mem_estimator, name='shaq'):
    plan = res['plan']
    partition_result = plan['partition_result']
    bit_assignment = plan['bit_assignment']
    D = res['D']
    prefill_bz = res['prefill_bz']
    bz_decode_max = res['bz_decode_max']
    bs_pack = (prefill_bz, bz_decode_max)
    # print("verify memory budget for", name)
    D_mem = get_device_topo_available_mem_with_order(D, model_mem_estimator, prefill_bz, bz_decode_max)
    for device_rank, layers_range in partition_result.items():
        device_mem = D_mem[device_rank]
        flag = check_memory_budget_single_device(device_mem, device_rank, layers_range, bit_assignment, \
                                           model_mem_estimator, bs_pack)
        if not flag:
            print("memory budget exceeded, return False", name)
            import pdb; pdb.set_trace()
            return False
    # print("all passed")
    return True


def log_result(result, name):
    print(f"{name} result: Minimax Lat {result}")



def algo_main():
    # check whether exists /opt/gurobi/ and file under file
    if not os.path.exists('/opt/gurobi/'):
        assert False, "Please install gurobi and put the license file under /opt/gurobi/"
    args = common_argparser()
    # device info
    # modelname and size
    model_name = args.model_name
    model_size = args.model_size
    device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers # [2, 3]
    device_info = get_device_info(device_names, device_numbers)
    args.device_info = device_info # use to store device info
    
    # run simulation
    global_bz = gen_config.global_bz
    micro_bz = gen_config.micro_bz
    s = gen_config.s
    n = gen_config.n
    model_size = args.model_size # '66b'
    device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers # [2, 3]
    gamma = _globals.gamma # expected generated tokens
    mu_n = int(gamma * n)
    # generation configs
    config = args.config
    comm_cost_model_dir = f'{args.comm_cost_model_dir}/{device_info}'
    cost_model_store_path = None
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
    
    # get solutions
    sol_adabits = adaptive_bits_main(args)
    # check how long shaq takes
    start = time.time()
    if args.shaq_efficient:
        # sol_shaq = shaq_ef_main(args)
        sol_shaq = shaq_h_main(args)
    else:
        sol_shaq = shaq_main(args)
    end = time.time()
    # sol_pipeedge_adaptive = pipeedge_adaptive_main(args)
    # sort by bit number, decsending
    no_info_bits = copy.deepcopy(_globals.AVAILABLE_BITS)[::-1]
    # no_info_bits.sort(reverse=True)
    # if args.adabits_tc:
    #     no_info_bits = copy.deepcopy(_globals.AVAILABLE_BITS)[::-1]
    

    # find first solution that is valid
    uniform_sols = {}
    for bit in no_info_bits:
        print("Try uniform bit: ", bit)
        args.uniform_bit = bit
        sol_uniform = uniform_main(args)
        if sol_uniform['plan'] != NOT_AVAILABLE:
            print("Uniform solution found, use bit: ", bit)
            uniform_sols[bit] = sol_uniform
            break

    # same to pipeedge
    for bit in no_info_bits:
        print("Try pipeedge bit: ", bit)
        args.pe_bit = bit
        sol_pipeedge = pipeedge_ilp_main(args)
        if sol_pipeedge['plan'] != NOT_AVAILABLE:
            print("PipeEdge solution found, use bit: ", bit)
            break
    # solution packs
    sols = {}
    sols['adabits'] = sol_adabits
    sols['shaq'] = sol_shaq
    sols['pipeedge'] = sol_pipeedge
    # sols['pipeedge_adaptive'] = sol_pipeedge_adaptive
    for bit, sol in uniform_sols.items():
        sols[f'uniform'] = sol
    for sol_name, sol in sols.items():
        print(f"start to run {sol_name}")
        if sol['plan'] == NOT_AVAILABLE:
            print(f"no plan for {sol_name}")
            continue
        check_memory_budget(sol, model_mem_estimator, name=sol_name)
        convert_to_shaq_result2partitions(sol)
        result = run_simu(gen_config, sol, lat_cost_model, comm_cost_model, \
                            args.use_profiler_prediction, mu_n=mu_n, comm_multiplier=args.comm_multiplier)
        
        log_result(result, sol_name)
    
    for sol_name, sol in sols.items():
        print("Minimum bit of ", sol_name)
        check_minimum_bit_of_sols(sol)

    # device info
    # pipedge device
    D_original = sol_pipeedge['D']
    D_shaq = sol_shaq['D']
    # check whether same, same key value
    for k, v in D_original.items():
        if D_shaq[k] != v:
            print("Shaq D not same")
            print(D_shaq)
            break
    
    # print(D_original)
    # print shaq time also
    print("Shaq time: ", end - start)
    sols['mu_n'] = mu_n
    sols['n'] = n
    sols['gloabl_bz'] = global_bz
    sols['prompt_length'] = s
    sols['model_name'] = model_name
    sols['model_size'] = model_size
    # store the solution
    # with device_names and model_name and model_size
    file_name = get_final_strat_file_name(model_name, model_size, device_info)
    if args.fname_suffix is not None:
        # insert before .pkl
        file_name = file_name[:-4] + args.fname_suffix + '.pkl'
    folder = args.store_folder
    save_with_pickle(sols, file_name, folder)
    logger.info(f'All plans saved to {file_name} in {folder}')

# if __name__ == '__main__':
#     main()