from qllm.utils import ModelMemEstimator
from ..cost_model import CommCostModel, LatCostModel

from transformers import (
    BloomConfig,
    OPTConfig,
)

def create_mem_estimator(b, s, n, config):
    vocab_size = config.vocab_size
    h1 = config.hidden_size
    if isinstance(config, OPTConfig):
        max_position_embeddings = config.max_position_embeddings
        word_embed_proj_dim = config.word_embed_proj_dim
        h2 = config.ffn_dim
    else:
        max_position_embeddings = 0
        word_embed_proj_dim = h1
        h2 = h1 * 4
    model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n, \
                                            vocab_size=vocab_size, max_position_embeddings=max_position_embeddings, word_embed_proj_dim=word_embed_proj_dim)
    return model_mem_estimator

# helper, init parameters and cost models
def init_parameters_and_cost_models(config, device_names=[], cost_model_store_path=None, \
                                     global_bz=16, micro_bz=4, prompt_length=512, num_token_to_generate=100, \
                                    comm_cost_model_folder='/workspace/qpipe/scripts/comm_cost_model/'):
    # target model configuration
    h1 = config.hidden_size
    h2 = config.ffn_dim
    num_hidden_layers = config.num_hidden_layers # decoder layer numbers

    b = global_bz
    s = prompt_length
    n = num_token_to_generate
    micro_b = micro_bz

    # T equals to num_hidden_layers, 0,1
    T = [0,1] * num_hidden_layers

    # estimators
    model_mem_estimator = create_mem_estimator(b, s, n, config)
    comm_size = (b * 1 * h1 * 2) / 1024 / 1024 # MB

    # cost models
    comm_cost_model = CommCostModel(comm_cost_model_folder=comm_cost_model_folder)
    if len(device_names) == 0:
        lat_cost_model = None
    else:
        lat_cost_model = LatCostModel(device_names, lat_cost_model_folder=cost_model_store_path)
        lat_cost_model.register_hyper_params(micro_b, s+n, h1, h2)
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



# result helper
def calculate_max_throughputs_and_lat(D, p_partition_result, p_bit_assign, \
                                       lat_cost_model, comm_cost_model, use_profiler_prediction=False, comm_size=0):
    minmax_lat = 0
    max_rank = len(D)
    for d_rank, device_name in D.items():
        if d_rank not in p_partition_result:
            continue
        layers_start, layers_end = p_partition_result[d_rank]
        comp_time = 0
        # latency
        for layer in range(layers_start, layers_end):
            shard = layer % 2
            bit = p_bit_assign[layer]
            if not use_profiler_prediction:
                comp_time += lat_cost_model.predict_with_hyper(device_name, shard, bit).item()
            else:
                lat = lat_cost_model.predict_with_profiled(device_name, shard, bit)
                try:
                    comp_time += lat
                except:
                    import pdb; pdb.set_trace()
        # communication
        next_rank = (d_rank + 1) % max_rank
        t_comm = comm_cost_model.predict_comm_time(d_rank, next_rank, comm_size)
        # minmax throughput
        minmax_lat = max(minmax_lat, comp_time, t_comm)
    # throughput
    tr = 1 / minmax_lat
    return minmax_lat, tr


from .utils import (
    create_device_mesh_grid,
)

def get_device_info(device_names, device_numbers):
    device_info = [f'{device_name}_{device_numbers[idx]}' for idx, device_name in enumerate(device_names)]
    device_info = '_'.join(device_info)
    return device_info

# help create device mesh
def create_device_mesh_and_mem(device_names, device_numbers):
    device_rank = []
    start_rank = 0
    for i in range(len(device_numbers)):
        device_rank.append(start_rank)
        start_rank += device_numbers[i]

    # create device mesh
    device_mesh = {device_rank[i]: [device_numbers[i], device_names[i]] for i in range(len(device_numbers))}

    D = create_device_mesh_grid(device_mesh)
    max_device_mem = get_device_mesh_overall_mem_constraints(D)
    return D, max_device_mem



# get SLO
from .._globals import SLO_RATE
from ..utils import query_bandwidth, get_device_mem_offline, partition_a_into_b_bins
from ..cost_model import estimate_all_layer_mem
import math 
from .utils import assign_uniform_bit
def get_slo(model_mem_estimator, comm_cost_model, lat_cost_model, T, comm_size, \
            device_names, use_profiler_prediction=True, verbose=True):
    num_hidden_layers = len(T) // 2
    # get the device with higher capability
    slo_rate = SLO_RATE
    bandwidth_m = 0 # the performance of decoding determined by bandwidth
    device_m = None
    device_mem = 0
    for device_name in device_names:
        bandwidth = query_bandwidth(device_name)
        if bandwidth > bandwidth_m:
            bandwidth_m = bandwidth
            device_m = device_name
            device_mem = get_device_mem_offline(device_name)
    # assign bits
    bit_map = {}
    assign_uniform_bit(T, 16, bit_map)
    initial_mem = estimate_all_layer_mem(model_mem_estimator, T, bit_map)
    num_gpus_required = math.ceil(initial_mem / device_mem)
    if verbose:
        print("SLO: use device: ", device_m)
        print("SLO: num_gpus_required: ", num_gpus_required)
    # create new device mesh
    device_names = [device_m]
    device_numbers = [num_gpus_required]
    D_SLO, _ = create_device_mesh_and_mem(device_names, device_numbers)
    # partition the layer evenly to the devices
    allocation = partition_a_into_b_bins(num_hidden_layers, num_gpus_required)
    # allocate
    partition_result = {}
    idx_start = 0
    for d_rank, device_name in D_SLO.items():
        layer_num = allocation[d_rank] * 2
        partition_result[d_rank] = [idx_start, idx_start + layer_num]
        idx_start += layer_num
    SLO_result = calculate_max_throughputs_and_lat(D_SLO, partition_result, bit_map, \
                                                        lat_cost_model, comm_cost_model, use_profiler_prediction, comm_size)
    lat = SLO_result[0]
    SLO_lat = slo_rate * lat
    if verbose:
        print("SLO_lat: ", SLO_lat, "lat: ", lat)
    return SLO_lat
