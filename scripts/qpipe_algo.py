# qpipe algorithm has a hyper parameter theta to determine the concern for the precision
# higher theta means more concern for the precision, lower theta means more concern for the latency/throughput
from qllm.models.OPT.opt import model_cards
from qllm.utils import ModelMemEstimator
from qpipe.partitioner import (
    assign_uniform_indicator
)
from qpipe.partitioner.utils import (
    assign_uniform_bit, 
    estimate_min_max_mem,
    get_maximum_available_mem,
    create_device_mesh_grid
)

from qpipe.partitioner.assigner import assign_bit_with_mem_constraints

from qpipe.cost_model import (
    estimate_all_layer_mem, estimate_single_layer_mem
)

from qpipe.utils import get_device_mem_offline

from qpipe.cost_model import CommCostModel, LatCostModel
from qpipe.utils import get_size_cpu
import qpipe
import torch
import pickle

unit = qpipe._globals.MEM_UNIT
RATIO_AVOID_OOM = qpipe._globals.RATIO_AVOID_OOM
CUDA_CONTEXT_MEM = qpipe._globals.CUDA_CONTEXT_MEM

# device configuration
device_names = ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
# device mesh
device_mesh = {
    0: [4, device_names[1]], # start rank, numbers, device_type
    4: [4, device_names[0]],
}

D = create_device_mesh_grid(device_mesh)

# get the maximum device memory
max_device_mem = get_maximum_available_mem(device_mesh)

# target model configuration
model_size = '175b'
config = model_cards[model_size]
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
lat_cost_model = LatCostModel(cost_model_store_path, device_names)
lat_cost_model.register_hyper_params(b, s+n, h1, h2)
# comm_cost_model.print_model_available_keys()
# comm_cost = comm_cost_model.predict_comm_time(start_rank=0, end_rank=1, data_size=get_size_cpu(x, unit='MB'))
# predicted_cost = lat_cost_model.predict(device, shard, b, i, h1, h2, bit)

file_name = 'qpipe_result.pkl' 
result_file_name = 'qpipe_result.txt'
available_bits = [2, 4, 8, '8:tc', '8:tc-li', 16] # we now can do hardware-aware quantization with 8:tc and 8:tc-li

def get_mem_available_devices(T, D, allocation_schemes, bit_assignment):
    if len(allocation_schemes) == 0: return D 
    # for each device, we need to estimate the memory usage
    est_mem_usage = {}
    for idx, device_rank in enumerate(allocation_schemes):
        shard = T[idx]
        bit = bit_assignment[idx]
        mem = estimate_single_layer_mem(model_mem_estimator, shard, bit)
        if device_rank not in est_mem_usage:
            est_mem_usage[device_rank] = mem
        else:
            est_mem_usage[device_rank] += mem
    
    # check memory usage for each device
    available_devices = {}
    for rank, device_name in enumerate(D):
        device_mem = RATIO_AVOID_OOM * get_device_mem_offline(device_name, unit=unit) - CUDA_CONTEXT_MEM
        if rank not in est_mem_usage:
            available_devices[rank] = device_mem
        else:
            if est_mem_usage[rank] < device_mem:
                available_devices[rank] = device_mem - est_mem_usage[rank]
    return available_devices

def transition_equation(min_max_throughput, shard, u, D, bit, allocation_schemes):
    # comp cost
    device_name = D[u]
    t_comp = lat_cost_model.predict_with_hyper(device_name, shard, bit).item()
    # comm cost
    # get last device in S
    t_comm = 0
    if len(allocation_schemes) != 0:
        # different from qpipe, the sequential order always ensures the previous rank's order
        # just get D's rank in order
        # rank_in_order = list(D.keys()).index(u)
        # last_device = D[rank_in_order - 1]
        t_comm = comm_cost_model.predict_comm_time(u-1, u, comm_size)
    res = max(min_max_throughput, t_comp, t_comm)
    return res 


def sequent_the_allocations(T, allocation_schemes):
    # just collect how many FFNs and attns for each device
    device_allocate_status = {}
    for layer_idx, device_rank in enumerate(allocation_schemes):
        if device_rank not in device_allocate_status:
            device_allocate_status[device_rank] = {'attn': 0, 'ffn': 0}
        shard = T[layer_idx]
        if shard == 0:
            device_allocate_status[device_rank]['attn'] += 1
        else:
            device_allocate_status[device_rank]['ffn'] += 1
    # all device with its allocated status
    # sort by ranks first
    device_allocate_status = {k: v for k, v in sorted(device_allocate_status.items(), key=lambda item: item[0])}
    # sequentially check whether the attn == ffn or abs( atten - FFN) == 1
    # if not, we need to adjust the allocation
    for device_rank, status in device_allocate_status.items():
        if abs(status['attn'] - status['ffn']) > 1:
            # handle start, move to the 

        else:
            continue



# qpipe didn't takes the precision information but search the precision with the dp process
def qpipe_algorithm(T, D):
    bit_assignment = {}
    assign_uniform_bit(T, 16, bit_assignment)
    L = len(T) # length of all the layers
    indicator = assign_uniform_indicator(L, available_bits)
    # we only allocate current layer to different devices.
    # then adjust them later.
    allocation_schemes = ()
    opt_throughput = 9999
    # TODO: 需要根据机器性能重排机器，因为只有这样才能确保每次分配都是最优的
    current_min_max_throughput = 0
    quant_step = 2 # each time, how many layers are quantized
    precision_ratio = 10 # the ratio determines how much we care about the precision or device
    # each time, the selection is optimal
    for i in range(L):
        avaialble_devices = get_mem_available_devices(T, D, allocation_schemes, bit_assignment)
        if len(avaialble_devices) > 0:
            select_device = None
            minimum_selcted_throughputs = 9999
            for device_rank, remaining_device_mem in avaialble_devices.items():
                # single algorithm, directly select the device that minimize the current layer
                shard = T[i]
                bit = bit_assignment[i]
                mem = estimate_single_layer_mem(model_mem_estimator, shard, bit)
                if mem < remaining_device_mem:
                    opt_throughput = transition_equation(current_min_max_throughput, shard, device_rank, D, bit, allocation_schemes)
                    if opt_throughput < minimum_selcted_throughputs:
                        minimum_selcted_throughputs = opt_throughput
                        select_device = device_rank
                    break
                else:
                    continue # the current device's memory is not enough, try next device
            if select_device is not None:
                allocation_schemes = allocation_schemes + (select_device, )
                current_min_max_throughput = minimum_selcted_throughputs

        if len(avaialble_devices) == 0 or select_device is None:
            # first handle the allocation schemes to make it sequenatial
            # that is, the allocation resulting in (1,2,3,2,1,) etc. but we want (1,1,2,2,3)
            # but also, by analyzing the FFN, the cost roughly 1.5x attn, so we try to exchange the result sequentially
            # do quantization to previous layers
            # select layer has 
            pass
    return bit_assignment
qpipe_algorithm(T, D)