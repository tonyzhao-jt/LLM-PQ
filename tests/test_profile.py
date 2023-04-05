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
    set_device_map(rank, device_mesh, sharding_strategy, rpc_opts)


if __name__ == '__main__':
    dist_cfg, device_mesh = init_env()

    # for test
    sharding_strategy = {
        0: {},
        1: {},
        2: {},
        3: {},
    }
    # run profile
    run_profile(sharding_strategy, dist_cfg, device_mesh)
