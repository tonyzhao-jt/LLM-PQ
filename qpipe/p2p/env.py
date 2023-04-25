import torch 
import os 

from .distconfig import DistConfig
def init_env():
    ngpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    group_rank = int(os.environ['GROUP_RANK'])
    # neighbor ranks
    dist_cfg = DistConfig(local_rank, rank, group_rank, world_size, ngpus)
    # torch set device
    torch.cuda.set_device(local_rank)
    return dist_cfg