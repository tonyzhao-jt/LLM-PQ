import os 
import copy 
import time 

# algorithms
from .adabits import main as adaptive_bits_main
from .shaq import main as shaq_main
from .shaq_efficient import main as shaq_ef_main
from .shaq_heuristic import main as shaq_h_main
from .pipeedge_ilp import main as pipeedge_ilp_main
from .uniform import main as uniform_main
from .lat_utils import run_simu
from .mem_utils import check_memory_budget
# arg parser
from .utils import (
    common_argparser,
    FP16_ENOUGH, NOT_AVAILABLE,
    get_final_strat_file_name,
    convert_to_shaq_result2partitions,
)
# globals
from .. import _globals 
from ..utils import save_with_pickle
from ..partitioner import gen_config
from ..partitioner.helper import (
    init_parameters_and_cost_models, 
    get_device_info,
)
# logger
from shaq.logger import logger, assert_log
from ..config import PROJECT_NAME

# for debug
def check_minimum_bit_of_sols(sol):
    bit_assignment = sol['plan']['bit_assignment']
    minimum_bit = 16
    for k, v in bit_assignment.items():
        if type(v) is str:
            v = 8
        if v < minimum_bit:
            minimum_bit = v
    logger.info("minimum_bit: ", minimum_bit)

def log_result(result, name):
    logger.info(f"{name} result: Minimax Lat {result}")

def algo_main():
    # check whether exists /opt/gurobi/ and file under file
    assert_log(os.path.exists('/opt/gurobi/'), "Please install gurobi and put the license file under /opt/gurobi/")
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
    no_info_bits = copy.deepcopy(_globals.AVAILABLE_BITS)[::-1]

    # find first solution that is valid
    uniform_sols = {}
    for bit in no_info_bits:
        logger.info("Try uniform bit: ", bit)
        args.uniform_bit = bit
        sol_uniform = uniform_main(args)
        if sol_uniform['plan'] != NOT_AVAILABLE:
            logger.info("Uniform solution found, use bit: ", bit)
            uniform_sols[bit] = sol_uniform
            break

    # same to pipeedge
    for bit in no_info_bits:
        logger.info("Try pipeedge bit: ", bit)
        args.pe_bit = bit
        sol_pipeedge = pipeedge_ilp_main(args)
        if sol_pipeedge['plan'] != NOT_AVAILABLE:
            logger.info("PipeEdge solution found, use bit: ", bit)
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
        logger.info(f"start to run {sol_name}")
        if sol['plan'] == NOT_AVAILABLE:
            logger.info(f"no plan for {sol_name}")
            continue
        check_memory_budget(sol, model_mem_estimator, name=sol_name)
        convert_to_shaq_result2partitions(sol)
        result = run_simu(gen_config, sol, lat_cost_model, comm_cost_model, \
                            args.use_profiler_prediction, mu_n=mu_n, comm_multiplier=args.comm_multiplier)
        
        log_result(result, sol_name)
    
    for sol_name, sol in sols.items():
        logger.info("Minimum bit of ", sol_name)
        check_minimum_bit_of_sols(sol)

    # device info
    # pipedge device
    D_original = sol_pipeedge['D']
    D_shaq = sol_shaq['D']
    # check whether same, same key value
    for k, v in D_original.items():
        if D_shaq[k] != v:
            logger.info(f"{PROJECT_NAME} D not same")
            logger.info(D_shaq)
            break
    
    logger.info(f"{PROJECT_NAME}: {end-start}")

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