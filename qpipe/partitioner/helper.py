from qllm.utils import ModelMemEstimator
from qllm.models import return_h1_h2
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
def init_parameters_and_cost_models(config, device_names=[], device_numbers=[], cost_model_store_path=None, \
                                     global_bz=16, micro_bz=4, prompt_length=512, num_token_to_generate=100, \
                                    comm_cost_model_folder='/workspace/qpipe/scripts/comm_cost_model/'):
    # target model configuration
    # h1 = config.hidden_size
    # h2 = config.ffn_dim
    h1, h2 = return_h1_h2(config)
    num_hidden_layers = config.num_hidden_layers # decoder layer numbers

    b = global_bz
    s = prompt_length
    n = num_token_to_generate
    micro_b = micro_bz

    # T equals to num_hidden_layers, 0,1
    T = [0,1] * num_hidden_layers

    # estimators
    model_mem_estimator = create_mem_estimator(b, s, n, config)
    # comm_size = (b * 1 * h1 * 2) / 1024 / 1024 # MB

    single_device = False
    if len(device_names) == 1 and len(device_numbers) == 1 and device_numbers[0] == 1:
        single_device = True
        print("is single device: ", device_names, device_numbers)
    # cost models
    comm_cost_model = CommCostModel(comm_cost_model_folder=comm_cost_model_folder, single_device=single_device)
    if len(device_names) == 0:
        lat_cost_model = None
    else:
        lat_cost_model = LatCostModel(device_names, lat_cost_model_folder=cost_model_store_path)
        lat_cost_model.register_hyper_params(micro_b, s, n, h1, h2)
    return model_mem_estimator, comm_cost_model, lat_cost_model, T


from .._globals import MEM_UNIT, RATIO_AVOID_OOM, CUDA_CONTEXT_MEM
from ..utils import get_device_mem_offline
def get_single_device_mem_constraints(device_name):
    device_name = device_name.upper()
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


def force_zero_3d(lat, z, prob):
    lat_shape = lat.shape
    for i in range(lat_shape[0]):
        for j in range(lat_shape[1]):
            for b in range(lat_shape[2]):
                if lat[i][j][b] > 999:
                    prob += z[(i, j, b)] == 0

def force_zero_2d(lat, z, prob):
    lat_shape = lat.shape
    for i in range(lat_shape[0]):
        for j in range(lat_shape[1]):
            if lat[i][j] > 999:
                prob += z[(i, j)] == 0
def force_zero(lat, z, prob):
    lat_shape = lat.shape
    if len(lat_shape) == 2:
        force_zero_2d(lat, z, prob)
    elif len(lat_shape) == 3:
        force_zero_3d(lat, z, prob)


def decouple_result_group(group_size, plan):
    # result: {i: (j, b)}, layer i place on device j withb
    # when i is a group, i is the first layer in the group
    new_plan = {}
    for i, (j, b) in plan.items():
        for k in range(group_size):
            new_plan[i * group_size+k] = (j, b) # set bit like that
    return new_plan




# produce latency prediction
def lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=True):
    stage_lat = 0
    if not use_profiler_prediction:
        atten_lat = lat_cost_model.predict_by_model_with_b_s_i_bit(D_name, 0, b, s, i, atten_bit)
        ffn_lat = lat_cost_model.predict_by_model_with_b_s_i_bit(D_name, 1, b, s, i, ffn_bit)
    else:
        atten_lat = lat_cost_model.predict_by_profiled_with_b_s_i_bit(D_name, 0, b, s, i, atten_bit)
        ffn_lat = lat_cost_model.predict_by_profiled_with_b_s_i_bit(D_name, 1, b, s, i, ffn_bit)
    
    if atten_lat is None:
        atten_lat = 9999
    if ffn_lat is None:
        ffn_lat = 9999
    stage_lat = (atten_lat + ffn_lat)
    return stage_lat

import numpy as np 
def get_latency_with_layer_device_bit_pair(current_D, bit_pairs, lat_cost_model, b, s, i, use_profiler_prediction=True):
    device_names = list(current_D.values())
    dtypes = set(device_names)
    device_bit_res = {}

    for device_name in dtypes:
        for idx, bit_pair in enumerate(bit_pairs):
            attn_bit, ffn_bit = bit_pair
            device_bit_res[(device_name, bit_pair)] = 0
            lat = lat_prediction(lat_cost_model, device_name, b, s, i, attn_bit, ffn_bit, use_profiler_prediction)
            device_bit_res[(device_name, bit_pair)] = lat
    # create latency matrix
    lat_device_bits_matrix = np.zeros((len(current_D), len(bit_pairs)))
    for i, device_name in enumerate(device_names):
        for j, bit_pair in enumerate(bit_pairs):
            lat_device_bits_matrix[i, j] = device_bit_res[(device_name, bit_pair)]
    return lat_device_bits_matrix