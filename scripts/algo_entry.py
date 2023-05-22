from adaptive_bits import main as adaptive_bits_main
from adaqpipe import main as adaqpipe_main
from pipeedge_ilp import main as pipeedge_ilp_main
from pipeedge_adaptive import main as pipeedge_adaptive_main
from uniform import main as uniform_main

# arg parser
from utils import (
    common_argparser, ilp_env,
    FP16_ENOUGH, NOT_AVAILABLE,
    get_final_strat_file_name
)
import qpipe
from qpipe.partitioner.helper import (
    get_device_info,
)
from qpipe.utils import save_with_pickle, has_tc
import logging
logger = logging.getLogger(__name__)

from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    lat_prediction,
    get_latency_with_layer_device_bit_pair
)
from qpipe.partitioner import gen_config
from qpipe.cost_model import (
    estimate_single_layer_mem,
)

import copy 
time_mult_times = qpipe._globals.TIME_MULT_TIMES
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
def check_memory_budget_single_device(device_rank, device_name, layers_range, bit_assignment, model_mem_estimator, bs_pack):
    i, j = layers_range
    prefill_bz, bz_decode_max = bs_pack
    # k % 2 means shard
    i_to_j_mem = sum([estimate_single_layer_mem(model_mem_estimator, 0, bit_assignment[k * 2]) for k in range(i, j)]) + \
        sum([estimate_single_layer_mem(model_mem_estimator, 1, bit_assignment[k * 2 + 1]) for k in range(i, j)])
    device_mem = get_single_device_mem_constraints(device_name)
    temp_tensor_mem = model_mem_estimator.calculate_temp_tensor_size_with_bz(prefill_bz, bz_decode_max, unit='MB')[0] 
    temp_later_decode = model_mem_estimator.calculate_temp_tensor_size_next_i(unit='MB')[0]

    if device_rank == 0:
        post_pre_mem = model_mem_estimator.calculate_prepost_mem(unit='MB')[0]
        device_mem = device_mem - post_pre_mem 
        device_mem -= (time_mult_times * temp_tensor_mem)
    else:
        device_mem -= (time_mult_times * temp_later_decode)

    
    if i_to_j_mem > device_mem:
        print(f"memory budget exceeded for device {device_rank}, {i_to_j_mem} > {device_mem}")
        return False
    return True

def check_memory_budget(res, model_mem_estimator, name='qpipe'):
    plan = res['plan']
    partition_result = plan['partition_result']
    bit_assignment = plan['bit_assignment']
    D = res['D']
    prefill_bz = res['prefill_bz']
    bz_decode_max = res['bz_decode_max']
    bs_pack = (prefill_bz, bz_decode_max)
    print("verify memory budget for", name)
    for device_rank, layers_range in partition_result.items():
        device_name = D[device_rank]
        flag = check_memory_budget_single_device(device_rank, device_name, layers_range, bit_assignment, \
                                           model_mem_estimator, bs_pack)
        if not flag:
            print("memory budget exceeded, return False", name)
            return False
    print("all passed")
    return True


def log_result(result, name):
    print(f"{name} result: Minimax Lat {result}")


# convert to the result can be used by adaqpipe
def convert_to_adaqpipe_result2partitions(res):
    pipeline_partition_result, bit_assignment_result = res['plan']['partition_result'], res['plan']['bit_assignment']
    D = res['D']
    # result is something like
    '''
        sharding_strategy = {
        0: {},
        1: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
        },
        2: {
            8: {'shard': [1], 'bits': [16]},
            9: {'shard': [0,1], 'bits': [16, 16]},
            10: {'shard': [0,1], 'bits': [8, 16]},
        },
    }
    '''
    sharding_strategy = {}
    for device_rank, (layer_start, layer_end) in pipeline_partition_result.items():
        sharding_strategy[device_rank] = {}
        for layer in range(layer_start, layer_end):
            atten_idx = layer * 2 
            ffn_idx = layer * 2 + 1
            atten_bit = bit_assignment_result[atten_idx]
            ffn_bit = bit_assignment_result[ffn_idx]
            # check if the bit can be replace with tc:8
            # TODO: abandon for the moment.
            # D_name = D[device_rank]
            # if atten_bit == 8:
            #     if has_tc(D_name):
            #         atten_bit = '8:tc'
            # if ffn_bit == 8:
            #     if has_tc(D_name):
            #         ffn_bit = '8:tc'
            sharding_strategy[device_rank][layer] = {
                'shard': [0, 1], 'bits': [atten_bit, ffn_bit]
            }

    res['use_plan'] = sharding_strategy


def calculate_max_stage_lat(D, use_plan, \
                                       cost_model_pack, b, s=1, i=1, use_profiler_prediction=False, comm_size=0):
    lat_cost_model, comm_cost_model = cost_model_pack

    minmax_lat = 0
    for device_rank, shard_strategy in use_plan.items():
        stage_lat = 0
        D_name = D[device_rank]
        for layer_idx, layer_spec in shard_strategy.items():
            shard = layer_spec['shard']
            bit = layer_spec['bits']
            atten_bit, ffn_bit = bit
            stage_lat += lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=use_profiler_prediction)
        # next stage
        next_stage = (device_rank + 1) % len(D)
        t_comm = comm_cost_model.predict_comm_time(device_rank, next_stage, comm_size * i)
        # minmax throughput
        minmax_lat = max(minmax_lat, stage_lat, t_comm)
    return minmax_lat

import math 
def run_simu(sol, lat_cost_model, comm_cost_model, use_profiler_prediction, comm_size, mu_n):
    D = sol['D']
    use_plan = sol['use_plan']
    prefill_bz = sol['prefill_bz']
    bz_decode_max = sol['bz_decode_max']
    maps = sol['maps']
    if maps is not None:
        comm_cost_model.set_device_rank_map(maps)
    global_bz = gen_config.global_bz
    data_pack = (prefill_bz, bz_decode_max)
    cost_model_pack = (lat_cost_model, comm_cost_model)
    s = gen_config.s
    n = gen_config.n
    # average throughput should equals to 
    prefill_result = calculate_max_stage_lat(D, use_plan, \
                                                    cost_model_pack, prefill_bz, s, 0, use_profiler_prediction, comm_size)
    decode_result = calculate_max_stage_lat(D, use_plan, \
                                                    cost_model_pack, bz_decode_max, 1, s + int(mu_n / 2), use_profiler_prediction, comm_size)
    # latency equals
    e2e_lat = math.ceil(global_bz / prefill_bz + 1) * prefill_result + \
          math.ceil(global_bz / bz_decode_max + 1) * decode_result * (mu_n-1)
    # remove maps
    if maps is not None:
        comm_cost_model.clear_device_rank_map() 
    return e2e_lat

def main(args):
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
    gamma = qpipe._globals.gamma # expected generated tokens
    mu_n = int(gamma * n)
    # generation configs
    config = args.config
    comm_cost_model_dir = f'{args.comm_cost_model_dir}/{device_info}'
    cost_model_store_path = None
    model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names, device_numbers, cost_model_store_path, \
                                                                                                     global_bz, micro_bz, s, n, \
                                                                                                  comm_cost_model_folder=comm_cost_model_dir)
    
    args.init_pack = (model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size)
    lat_cost_model.update_profiled_result(args.lat_profile_dir)
    if args.fit:
        lat_cost_model.fit_regression_cost_model()
    if not args.use_profiler_prediction:
        lat_cost_model.load_regression_cost_model()
    
    # get solutions
    sol_adabits = adaptive_bits_main(args)
    sol_adaqpipe = adaqpipe_main(args)
    # sol_pipeedge_adaptive = pipeedge_adaptive_main(args)
    # sort by bit number, decsending
    no_info_bits = copy.deepcopy(qpipe._globals.AVAILABLE_BITS)[::-1]
    # no_info_bits.sort(reverse=True)
    # if args.adabits_tc:
    #     no_info_bits = copy.deepcopy(qpipe._globals.AVAILABLE_BITS)[::-1]
    

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
    sols['adaqpipe'] = sol_adaqpipe
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
        convert_to_adaqpipe_result2partitions(sol)
        result = run_simu(sol, lat_cost_model, comm_cost_model, \
                            args.use_profiler_prediction, comm_size=comm_size, mu_n=mu_n)
        
        log_result(result, sol_name)
    
    for sol_name, sol in sols.items():
        print("Minimum bit of ", sol_name)
        check_minimum_bit_of_sols(sol)

    print(sols['adaqpipe']['D'])
    sols['mu_n'] = mu_n
    sols['n'] = n
    sols['gloabl_bz'] = global_bz

    import pdb; pdb.set_trace()
    # store the solution
    # with device_names and model_name and model_size
    file_name = get_final_strat_file_name(model_name, model_size, device_info)
    folder = args.store_folder
    save_with_pickle(sols, file_name, folder)
    logger.info(f'All plans saved to {file_name} in {folder}')

if __name__ == '__main__':
    args = common_argparser()
    main(args)