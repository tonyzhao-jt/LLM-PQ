import torch
import torch.distributed as dist 
import torch.multiprocessing as mp
import os 
from shaq.logger import logger
from .tp import tp_launcher
from .global_context import *
def launch(rank, world_size, backend="nccl", tp_group=[], pp_group=[]):
    host = MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")
    port = MASTER_PORT = os.environ.get("MASTER_PORT", "12355")
    init_method = f'tcp://[{host}]:{port}'
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    logger.info("INIT NCCL WITH ADDRESS %s:%d" % (MASTER_ADDR, MASTER_PORT))

    global_rank = dist.get_rank()
    n_gpus = torch.cuda.device_count()
    n_gpus = torch.cuda.device_count()
    local_size = n_gpus // mp.get_world_size()
    local_rank = global_rank % local_size

    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(global_rank)

    # set by qllm 
    if len(tp_group) != 0:
        tp_launcher(tp_group, backend=backend)
    
    # set pp group
    if len(pp_group) != 0:
        set_pp_group_ranks(pp_group)
    
    