from qllm.utils import ModelMemEstimator
from ..cost_model import CommCostModel, LatCostModel

# helper, init parameters and cost models
def init_parameters_and_cost_models(config, device_names=[]):
    # target model configuration
    h1 = config.hidden_size
    h2 = config.ffn_dim
    num_hidden_layers = config.num_hidden_layers # decoder layer numbers

    # micro_batch_size
    b = 16
    # set the prompt sequence length
    s = 512
    # set the number of generated tokens
    n = 100

    # T equals to num_hidden_layers, 0,1
    T = [0,1] * num_hidden_layers

    # estimators
    model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n)
    comm_size = (b * 1 * h1 * 2) / 1024 / 1024 # MB

    # cost models
    cost_model_store_path = '/workspace/qpipe/scripts/lat_cost_model'
    comm_cost_model = CommCostModel(comm_cost_model_folder='/workspace/qpipe/scripts/comm_cost_model/')
    if len(device_names) == 0:
        lat_cost_model = None
    else:
        lat_cost_model = LatCostModel(cost_model_store_path, device_names)
        lat_cost_model.register_hyper_params(b, s+n, h1, h2)
    return model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size


from .._globals import MEM_UNIT, RATIO_AVOID_OOM, CUDA_CONTEXT_MEM
from ..utils import get_device_mem_offline
def get_single_device_mem_constraints(device_name):
    device_mem = RATIO_AVOID_OOM * get_device_mem_offline(device_name, unit=MEM_UNIT) - CUDA_CONTEXT_MEM
    return device_mem

def get_device_mesh_overall_mem_constraints(D):
    overall_mem = 0
    for device_rank, device_name in D.items():
        device_mem = get_single_device_mem_constraints(device_name)
        overall_mem += device_mem
    return overall_mem