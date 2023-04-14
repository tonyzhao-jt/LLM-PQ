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



