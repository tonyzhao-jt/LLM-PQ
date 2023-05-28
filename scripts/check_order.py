from utils import common_argparser, get_final_strat_file_name
import pickle
import qpipe
from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info,
    lat_prediction,
    get_latency_with_layer_device_bit_pair
)
from algo_entry import check_memory_budget
args = common_argparser()
model_name = args.model_name
model_size = args.model_size
device_info = args.device_info
file_name = get_final_strat_file_name(model_name, model_size, device_info)
folder = args.store_folder
abs_file_name = folder + '/' + file_name
with open(abs_file_name, 'rb') as f:
    sols = pickle.load(f)

test_method = args.test_method
sol = sols[test_method]
D = sol['D']
print(D)
print(device_info)
print(sol)

# run simulation
from qpipe.partitioner import gen_config
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
model_mem_estimator, comm_cost_model, lat_cost_model, T = init_parameters_and_cost_models(config, device_names, device_numbers, cost_model_store_path, \
                                                                                                     global_bz, micro_bz, s, n, \
                                                                                                  comm_cost_model_folder=comm_cost_model_dir)


check_memory_budget(sol, model_mem_estimator, name=test_method)