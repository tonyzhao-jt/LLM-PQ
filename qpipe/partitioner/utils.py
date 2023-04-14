def create_device_mesh_grid(device_mesh):
    # based on the device_mesh, create D
    D = {}
    for start_rank, single_node_device in device_mesh.items():
        num_devices, device_type = single_node_device
        for i in range(num_devices):
            D[start_rank + i] = device_type
    return D

# assign bits to the layer
def assign_uniform_bit(layers, bit, bit_map):
    for idx, layer in enumerate(layers):
        bit_map[idx] = bit
    return bit_map

from ..cost_model import estimate_all_layer_mem, estimate_single_layer_mem
def estimate_min_max_mem(estimator, layers, max_bit=16, min_bit=2):
    bit_map = {}
    assign_uniform_bit(layers, max_bit, bit_map)
    max_mem = estimate_all_layer_mem(estimator, layers, bit_map)
    assign_uniform_bit(layers, min_bit, bit_map)
    min_mem = estimate_all_layer_mem(estimator, layers, bit_map)
    return max_mem, min_mem

def get_bit_layer_memory_map(estimator, available_bits, shards=[0,1]):
    s = {}
    for shard in shards:
        for bit in available_bits:
            mem_require = estimate_single_layer_mem(estimator, shard, bit)
            s[(shard, bit)] = mem_require
    return s 

def get_bitidx_layer_memory_map(estimator, available_bits, shards=[0,1]):
    s = {}
    for shard in shards:
        for idx, bit in enumerate(available_bits):
            mem_require = estimate_single_layer_mem(estimator, shard, bit)
            s[(shard, idx)] = mem_require
    return s 

def interpret_ilp_result_i_j_b(ilp_result, available_bits):
    # qpipe result
    # handle result into pipeegde form
    pipeline_partition_result_qpipe = {}
    bit_assignment_result_qpipe = {}
    L = len(list(ilp_result.keys()))
    for layer, (device_rank, bit) in ilp_result.items():
        if device_rank not in pipeline_partition_result_qpipe:
            pipeline_partition_result_qpipe[device_rank] = []
            bit_assignment_result_qpipe[device_rank] = []
        pipeline_partition_result_qpipe[device_rank].append(layer)
        bit_assignment_result_qpipe[device_rank].append(bit)

    # reset the layer index
    for device_rank, layers in pipeline_partition_result_qpipe.items():
        pipeline_partition_result_qpipe[device_rank] = len(layers)
    start_idx = 0
    for device_rank, layers in pipeline_partition_result_qpipe.items():
        pipeline_partition_result_qpipe[device_rank] = [start_idx, start_idx + layers * 2]
        start_idx += layers * 2

    pipeline_partition_result_qpipe = {k: pipeline_partition_result_qpipe[k] for k in sorted(pipeline_partition_result_qpipe)}

    available_bits = list(set(available_bits))
    BITs = [
        (i, j) for i in available_bits for j in available_bits
    ]
    # assign bits
    new_bit_assignment_result_qpipe = {}
    for device_rank, bit in bit_assignment_result_qpipe.items():
        part_result = pipeline_partition_result_qpipe[device_rank]
        bit_idx = 0
        for i in range(part_result[0], part_result[1], 2):
            attn_bit, ffn_bit = BITs[bit[bit_idx]]
            new_bit_assignment_result_qpipe[i] = attn_bit
            new_bit_assignment_result_qpipe[i+1] = ffn_bit
            bit_idx += 1
    bit_assignment_result_qpipe = new_bit_assignment_result_qpipe
    return {
        'partition_result': pipeline_partition_result_qpipe,
        'bit_assignment': bit_assignment_result_qpipe 
    }
