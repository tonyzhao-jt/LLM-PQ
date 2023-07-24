import argparse
from qllm.models import create_model_config
from ..partitioner.helper import (
    get_device_info,
)
from ..partitioner import gen_config
from .. import _globals 
import os

ROOT_DIR = os.environ.get('ROOT_DIR', '/workspace/shaq')
def verbose_device_info(device_names, device_numbers, device_info):
    print(f"device_names {device_names}")
    print(f"device_numbers {device_numbers}")
    print(f"device_info {device_info}")

def common_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='opt')
    parser.add_argument('--model_size', type=str, required=True)
    parser.add_argument('--device_names',  nargs='+', type=str, required=True)
    parser.add_argument('--device_numbers',  nargs='+', type=int, required=True)
    parser.add_argument('--SLO-aware',  action='store_true', help='add slo into constraints')
    parser.add_argument('--omega_file', type=str, default=None)
    parser.add_argument('--use_profiler_prediction', action='store_true', help='use profiler prediction')
    parser.add_argument('--comm_cost_model_dir', type=str, default=f'{ROOT_DIR}/scripts/comm_cost_model/')
    parser.add_argument('--lat_profile_dir', type=str, default=f'{ROOT_DIR}/scripts/lat_profiled_result')
    parser.add_argument('--lat_prepost_profile_dir', type=str, default=f'{ROOT_DIR}/scripts/lat_prepost_profiled_result')
    parser.add_argument('--store_folder', type=str, default=f'{ROOT_DIR}/scripts/part_strategy')
    # ilp control
    # different seed result in different performance
    parser.add_argument('--ilp_seed', type=int, default=42)
    parser.add_argument('--group_size', type=int, default=1) # when search space is too large, need to group
    parser.add_argument('--ilp_tolerance', type=float, default=None)
    parser.add_argument('--ilp_time_limit', type=int, default=None)
    parser.add_argument('--adapp_group_size', type=int, default=1)
    # algo control
    parser.add_argument('--pe_bit', type=int, default=8)
    parser.add_argument('--uniform_bit', type=int, default=8)
    parser.add_argument('--adabits_tc', action='store_true', help='use adabit-tc') # case when all device support tc
    parser.add_argument('--init_pack', default=None)
    parser.add_argument('--uniform-hybrid', default=True)
    # experiment setup control
    parser.add_argument('--s', type=int, default=512) # prompt legnth
    parser.add_argument('--n', type=int, default=100) # max_tokens
    parser.add_argument('--global_bz', type=int, default=16) # global batch size
    parser.add_argument('--theta', type=float, default=0.0001) # concern for accuracy
    parser.add_argument('--gamma', type=float, default=0.8) # expected token numbers (x max) to generate
    parser.add_argument('--comm_multiplier', type=float, default=1) # multiply communication when not only the hidden space is passed.
    parser.add_argument('--time_mult_times', type=float, default=1)
    # for debug and fit
    parser.add_argument('--fit', action='store_true', help='fit cost model')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--fname', type=str, default=None)
    # choose from 'adabits' 'adaqpipe' 'pipeedge' 'uniform'
    parser.add_argument('--test_method', type=str, default='adabits', help='test method')
    # storage control
    parser.add_argument('--fname-suffix', type=str, default=None) # add suffix to generated solution plan
    args = parser.parse_args()

    # temporary memory control
    _globals.TIME_MULT_TIMES = args.time_mult_times

    # modelname and size
    model_name = args.model_name
    model_size = args.model_size
    config = create_model_config(model_name, model_size)
    args.config = config

    # set configs
    gen_config.global_bz = args.global_bz
    gen_config.s = args.s
    gen_config.n = args.n
    _globals.gamma = args.gamma
    _globals.theta = args.theta

    # checks
    device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers # [2, 3]
    device_info = get_device_info(device_names, device_numbers)
    args.device_info = device_info
    assert len(device_names) == len(device_numbers), f"device_names and device_numbers \
          should have the same length {device_names} {device_numbers}"

    if args.debug:
        verbose_device_info(args.device_names, args.device_numbers, device_info)
    
    # check omega file valid if exits
    if args.omega_file is not None:
        assert os.path.exists(args.omega_file), f"omega file {args.omega_file} does not exist"
        assert model_name in args.omega_file, f"omega file {args.omega_file} does not contain model name {model_name}"
        assert model_size in args.omega_file, f"omega file {args.omega_file} does not contain model size {model_size}"

    return args

import pulp
import gurobipy as gp
import os
def ilp_env(timeLimit=None):
    # check whether file exists under 
    path = "/opt/gurobi/"
    if not os.path.exists(path):
        raise Exception("Gurobi is not installed")
    env = gp.Env(empty=True)
    env.setParam('WLSACCESSID',"1b28dca7-337e-4811-b346-01087e09cd64")
    env.setParam('WLSSECRET', "629520bd-a114-45d7-b828-bfc5235c198d")
    env.setParam('LICENSEID', 965996)
    if timeLimit is not None:
        env.setParam('TimeLimit', timeLimit)
    env.start()

SIGNAL_BASE=1234
NOT_AVAILABLE=SIGNAL_BASE + 1
FP16_ENOUGH=SIGNAL_BASE + 2

def get_final_strat_file_name(model_name, model_size, device_info):
    file_name = f'sols_' + f'{model_name}_{model_size}' + '_' + device_info + '.pkl'
    return file_name




# processing
# convert to the result can be used by shaq
def convert_to_shaq_result2partitions(res):
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


def create_ilp_solver(verbose_ilp, ilp_time_limit, ilp_tolerance):
    args = {"msg": verbose_ilp, "timeLimit": ilp_time_limit, "MIPGap": ilp_tolerance}
    if ilp_tolerance is None:
        args.pop("MIPGap")
    if ilp_time_limit is None:
        args.pop("timeLimit")
    solver = pulp.GUROBI(**args)
    return solver