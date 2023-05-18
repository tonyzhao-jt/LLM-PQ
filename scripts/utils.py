import argparse
from qllm.models import create_model_config
def verbose_device_info(device_names, device_numbers):
    from qpipe.partitioner.helper import (
        get_device_info,
    )
    print(f"device_names {device_names}")
    print(f"device_numbers {device_numbers}")
    device_info = get_device_info(device_names, device_numbers)
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
    parser.add_argument('--comm_cost_model_dir', type=str, default='/workspace/qpipe/scripts/comm_cost_model/')
    parser.add_argument('--lat_profile_dir', type=str, default='/workspace/qpipe/scripts/lat_profiled_result')
    # for pipeedge
    parser.add_argument('--pe_bit', type=int, default=8)
    parser.add_argument('--uniform_bit', type=int, default=8)
    # experiment setup control
    parser.add_argument('--s', type=int, default=512) # prompt legnth
    parser.add_argument('--n', type=int, default=100) # max_tokens
    parser.add_argument('--global_bz', type=int, default=16) # global batch size
    parser.add_argument('--theta', type=float, default=0.01) # concern for accuracy
    parser.add_argument('--gamma', type=float, default=0.8) # expected token numbers (x max) to generate

    # for debug and control
    parser.add_argument('--fit', action='store_true', help='fit cost model')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()


    # modelname and size
    model_name = args.model_name
    model_size = args.model_size
    config = create_model_config(model_name, model_size)
    args.config = config


    # check omega file valid if exits
    if args.omega_file is not None:
        assert os.path.exists(args.omega_file), f"omega file {args.omega_file} does not exist"
        assert model_name in args.omega_file, f"omega file {args.omega_file} does not contain model name {model_name}"
        assert model_size in args.omega_file, f"omega file {args.omega_file} does not contain model size {model_size}"

    # checks
    device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
    device_numbers = args.device_numbers # [2, 3]
    assert len(device_names) == len(device_numbers), f"device_names and device_numbers \
          should have the same length {device_names} {device_numbers}"

    if args.debug:
        verbose_device_info(args.device_names, args.device_numbers)
    return args

import pulp
import gurobipy as gp
import os
def ilp_env():
    # check whether file exists under 
    path = "/opt/gurobi/"
    if not os.path.exists(path):
        raise Exception("Gurobi is not installed")
    env = gp.Env(empty=True)
    env.setParam('WLSACCESSID',"1b28dca7-337e-4811-b346-01087e09cd64")
    env.setParam('WLSSECRET', "629520bd-a114-45d7-b828-bfc5235c198d")
    env.setParam('LICENSEID', 965996)
    env.start()

SIGNAL_BASE=1234
NOT_AVAILABLE=SIGNAL_BASE + 1
FP16_ENOUGH=SIGNAL_BASE + 2
