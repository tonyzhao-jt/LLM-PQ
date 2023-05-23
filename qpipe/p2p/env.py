import torch 
import os 

from .distconfig import DistConfig
from .device import create_device_mesh, create_device_mesh_nccl
def init_env():
    ngpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    group_rank = int(os.environ['GROUP_RANK'])
    # neighbor ranks
    dist_cfg = DistConfig(local_rank, rank, group_rank, world_size, ngpus)
    # hard_device_mesh = create_device_mesh_nccl(rank, local_rank, world_size)
    # torch set device
    torch.cuda.set_device(local_rank)
    return dist_cfg, None


def new_nccl_group():
    world_size = int(os.environ['WORLD_SIZE'])
    # new a same group with the same worldsize
    group = torch.distributed.new_group(list(range(world_size)), backend='nccl')
    return group