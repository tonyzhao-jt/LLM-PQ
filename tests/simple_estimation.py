from shaq.partitioner.helper import init_parameters_and_cost_models
from qllm.models.OPT.opt import model_cards
device_names = ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
model_size = '350M'
config = model_cards[model_size]
model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size = init_parameters_and_cost_models(config, device_names)

from shaq.cost_model import (
    estimate_all_layer_mem, estimate_single_layer_mem
)
mem_1 = estimate_single_layer_mem(model_mem_estimator, 1, 16)
mem_2 = estimate_single_layer_mem(model_mem_estimator, 0, 16)
print(9 * (mem_1 + mem_2))