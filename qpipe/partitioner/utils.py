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

# layer device bit
def interpret_ilp_result_i_j_b(ilp_result, BITs):
    device_layer_dict = {}
    layer_to_bit_map = {}
    for layer, (device_rank, bit_idx) in ilp_result.items():
        bit_pair = BITs[bit_idx]
        if device_rank not in device_layer_dict:
            device_layer_dict[device_rank] = [layer]
            layer_to_bit_map[device_rank] = [bit_pair]
        else:
            device_layer_dict[device_rank].append(layer)
            layer_to_bit_map[device_rank].append(bit_pair)
    
    partition_result = {}
    start = 0
    # sort device_layer_dict by device_rank
    device_layer_dict = {k: device_layer_dict[k] for k in sorted(device_layer_dict)}
    # sort the partition among layers.
    for device_rank, layers in device_layer_dict.items():
        partition_result[device_rank] = [start, start + len(layers)]
        start += len(layers)
    # generate bitwidth mapping
    bit_assignment_result = {}
    for device_rank, (layer_start, layer_end) in partition_result.items():
        bit_pairs = layer_to_bit_map[device_rank]
        bit_pair_idx = 0
        for layer in range(layer_start, layer_end):
            attn_layer = layer * 2
            ffn_layer = layer * 2 + 1
            bit_pair = bit_pairs[bit_pair_idx]
            attn_bit, ffn_bit = bit_pair
            # map
            bit_assignment_result[attn_layer] = attn_bit
            bit_assignment_result[ffn_layer] = ffn_bit
            bit_pair_idx += 1
    
    return {
        'partition_result': partition_result,
        'bit_assignment': bit_assignment_result
    }
