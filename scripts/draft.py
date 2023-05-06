
from qpipe.partitioner.utils import (
    assign_uniform_bit, 
)

from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_single_device_mem_constraints,
    create_device_mesh_and_mem,
    get_device_info
)

from qpipe.partitioner.helper import (
    init_parameters_and_cost_models, 
    get_slo
)
from qllm.models.OPT.opt import model_cards

# help input the config
from qpipe.partitioner import gen_config
# generation configs
global_bz = gen_config.global_bz
micro_bz = gen_config.micro_bz
s = gen_config.s
n = gen_config.n

model_size = '30b'

device_names = ['Tesla_T4', 'Tesla_V100-SXM2-32GB']
device_numbers = [1, 2]

config = model_cards[model_size]
D, max_device_mem = create_device_mesh_and_mem(device_names, device_numbers)
# max_device_mem can be used to check whether OOM or not
use_profiler_prediction = True
# target model configuration
cost_model_store_path = None # initialize the cost model
model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names, cost_model_store_path, \
                                                                                                     global_bz, micro_bz, s, n)
num_hidden_layers = len(T) // 2
num_devices = len(D)

if use_profiler_prediction:
    lat_cost_model.update_profiled_result('/workspace/qpipe/scripts/lat_profiled_result')

SLO_lat = get_slo(model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size, \
            device_names, use_profiler_prediction=True, verbose=True)
print(SLO_lat)

import pdb; pdb.set_trace()