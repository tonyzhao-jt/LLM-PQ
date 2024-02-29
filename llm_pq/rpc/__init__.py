from .distconfig import DistConfig
from .device import create_device_mesh, create_device_mesh_nccl, set_device_map, get_neighbor_ranks, get_local_rank_by_device_mesh

import os 
import torch
from typing import Any, Callable, List, Optional, Tuple, Type, Union
from torch.distributed import rpc
import torch.distributed as dist
from .comm import stop_event
from .utils import ConditionQueue
# RPC CONTEXT
DistCmdHandler: Type = Callable[[int, Tuple[torch.Tensor, ...]], None]

class DistContext:
    """Parent class for distributed context managers."""

    def __init__(self, init_args: tuple, init_kwargs: dict):
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._world_size = init_kwargs['world_size']
        self._rank = init_kwargs['rank']
        self._initialized = False

    def init(self) -> None:
        """Initialize the distributed context."""
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the distributed context."""
        self._initialized = False

    def __enter__(self):
        assert not self._initialized
        self.init()
        return self

    def __exit__(self, *args):
        assert self._initialized
        self.shutdown()

class DistRpcContext(DistContext):
    """The singleton distributed RPC context manager."""

    def init(self) -> None:
        """Initialize the distributed context."""
        super().init()
        rpc.init_rpc(*self._init_args, **self._init_kwargs)

    def shutdown(self) -> None:
        """Wait for all RPCs to finish and shutdown the distributed context."""
        super().shutdown()
        rpc.shutdown()

    def cmd_broadcast(self, remote_cmd_handler: DistCmdHandler, cmd: int,
                      tensors: Optional[Tuple[torch.Tensor, ...]]=None) -> None:
        """Broadcast a command."""
        assert self._initialized
        if tensors is None:
            tensors = ()
        futs = []
        for rank in range(self._world_size):
            if rank != self._rank:
                fut = rpc.rpc_async(rank, remote_cmd_handler, args=(cmd, tensors))
                futs.append(fut)
        torch.futures.wait_all(futs)

def init_env_gloo():
    ngpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    group_rank = int(os.environ['GROUP_RANK'])
    # neighbor ranks
    torch.cuda.set_device(local_rank)
    hard_device_mesh = create_device_mesh(rank, local_rank, world_size)
    dist_cfg = DistConfig(local_rank, rank, group_rank, world_size, ngpus)
    return dist_cfg, hard_device_mesh

def init_env():
    ngpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    group_rank = int(os.environ['GROUP_RANK'])
    # neighbor ranks
    torch.cuda.set_device(local_rank)
    hard_device_mesh = create_device_mesh_nccl(rank, local_rank, world_size)
    dist_cfg = DistConfig(local_rank, rank, group_rank, world_size, ngpus)
    return dist_cfg, hard_device_mesh

# contruct rpc opt by environ
def rpc_opt_factory(rpc_timeout=60, rpc_disable_shm=False):
    import random
    from llm_pq.logger import logger
    MASTER_PORT = int(os.environ['MASTER_PORT'])
    if 'RPC_PORT' in os.environ:
        rpc_port = int(os.environ['RPC_PORT'])
    else:
        # randomly assign an rpc port
        # rpc_port = MASTER_PORT + 10 # set a different number
        rpc_port = MASTER_PORT
    rpc_options = {}
    if rpc_disable_shm:
        rpc_options['_transports'] = ['uv']
    master_host = os.environ['MASTER_ADDR']
    rpc_opts = rpc.TensorPipeRpcBackendOptions(
        rpc_timeout=rpc_timeout, init_method=f'tcp://{master_host}:{rpc_port}', **rpc_options)
    logger.info("INIT RPC WITH ADDRESS %s:%d" % (master_host, rpc_port))
    
    return rpc_opts


