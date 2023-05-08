from torch.distributed import rpc
from qpipe.rpc import get_local_rank_by_device_mesh, get_neighbor_ranks
import numpy as np 
from qpipe.logger import logger
from qpipe.rpc import (
    init_env, 
    DistConfig, set_device_map,
    DistRpcContext,
    stop_event
)
# the device_mesh deterimes the pp and tp order of execution
def set_device_map(cur_rank, device_mesh, hard_device_mesh, rpc_options):
    # check whether hard_device mesh is in the right format
    for first_rank, ranks in hard_device_mesh.items():
        assert first_rank == ranks[0], "hard_device_mesh should be in the format of {first_rank: [all_ranks on the same node]}"
    stages_2d = np.array(list(device_mesh.values())).T
    tp_group_size, stage_nums = stages_2d.shape
    # create device_map
    device_maps = {}
    # start from each row
    for j in range(stage_nums):
        # logger.info(f"stage {j} COMM Initialization")
        # iterate columns
        ranks_within_stage = stages_2d[:, j]
        next_stage = (j + 1) % stage_nums
        for i, rank in enumerate(ranks_within_stage):
            rank = int(rank)
            if rank != cur_rank:
                continue
            local_rank = get_local_rank_by_device_mesh(hard_device_mesh, rank)
            neighbor_ranks = get_neighbor_ranks(hard_device_mesh, rank)
            # for the time being, the communication between gpus on the same stage is done by p2p groups
            # so we don't need to set device map for them
            next_rank = int(stages_2d[i, next_stage])
            next_rank_local_rank = get_local_rank_by_device_mesh(hard_device_mesh, next_rank)
            # set map
            device_maps[f"worker{rank}"] = {local_rank: next_rank_local_rank}
            rpc_options.set_device_map(f"worker{rank}", {local_rank: next_rank_local_rank})
            logger.info(f"set device map for worker{rank}:{local_rank} to worker{next_rank}:{next_rank_local_rank}")
    return stages_2d, rpc_options

device_mesh = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
}

dist_cfg, hard_device_mesh = init_env()
rank = dist_cfg.rank
# hardware_device_mesh = {first_rank: [all_ranks on the same node]}
hard_device_mesh = {0: [0, 1, 2, 3, 4, 5, 6, 7]}
hard_device_mesh = {0: [0, 1, 2, 3], 4: [4, 5, 6, 7]}
rpc_opts = rpc.TensorPipeRpcBackendOptions(rpc_timeout=60) # the loading of weight takes a lot of time
set_device_map(rank, device_mesh, hard_device_mesh, rpc_opts)
with DistRpcContext((f"worker{rank}",),
                    { 'world_size': dist_cfg.world_size,
                        'rank': rank,
                        'rpc_backend_options': rpc_opts}
                    ) as dist_ctx:
    print("done")