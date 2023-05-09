from torch.distributed import rpc
from qpipe.rpc import get_local_rank_by_device_mesh, get_neighbor_ranks
import numpy as np 
from qpipe.logger import logger
from qpipe.rpc import (
    init_env, 
    DistConfig, set_device_map,
    DistRpcContext,
    stop_event,
    rpc_opt_factory
)


dist_cfg, hard_device_mesh = init_env()
rpc_opts = rpc_opt_factory(rpc_timeout=60, rpc_disable_shm=True) # the loading of weight takes a lot of time
rank = dist_cfg.rank


with DistRpcContext((f"worker{rank}",),
                    { 'world_size': dist_cfg.world_size,
                        'rank': rank,
                        'rpc_backend_options': rpc_opts}
                    ) as dist_ctx:
    print("done")

# rpc.init_rpc(f"worker{rank}", rank=rank, world_size=dist_cfg.world_size, rpc_backend_options=rpc_opts)