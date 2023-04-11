'''
    The profiler check the availability of the module on bitwidth but also profile the latency on them
    model_machine_layer_profile_result.pkl
    inside the profile result, we only consider selfattn and FFN
    1. Latency map.
    2. available bit. 
'''
from torch.distributed import rpc
from qpipe.rpc import init_env, DistConfig, set_device_map, DistRpcContext

def run_profile(sharding_strategy, dist_cfg: DistConfig, device_mesh):
    rpc_opts = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128, rpc_timeout=60)
    rank = dist_cfg.rank
    data_rank = 0 
    # contruct the device map
    # set_device_map(rank, device_mesh, sharding_strategy, rpc_opts)
    # for the communication cost in each line, we only need to profile the first node
    # for the communication from the end node of the line and the start node of the next line, we only need to profile the end node
    # thus, for each row, two nodes are enough. (for last row, one node is enough)
    all_test_pairs = [] # for ref
    last_node = None
    mesh_comm_cost = {}
    row_idx = 0
    for node_first_rank, ranks in device_mesh.items():
        # first two nodes
        all_test_pairs.append(tuple(ranks[:2]))
        mesh_comm_cost[f'node_{row_idx}_row_cost'] = tuple(ranks[:2])
        if last_node is not None:
            all_test_pairs.append((last_node, ranks[0]))
            mesh_comm_cost[f'node_{row_idx}_col_cost'] = (last_node, ranks[0])
        last_node = ranks[-1]
    
    with DistRpcContext((f"worker{rank}",),
                            { 'world_size': dist_cfg.world_size,
                            'rank': rank,
                            'rpc_backend_options': rpc_opts}
                        ) as dist_ctx:
    
        # do communication here



if __name__ == '__main__':
    dist_cfg, device_mesh = init_env()

    # for test
    sharding_strategy = {
        0: {},
        1: {},
        2: {},
        3: {},
    }

    test_tensor_size = (1, 7, 768)
    # run profile
    run_profile(sharding_strategy, dist_cfg, device_mesh)
