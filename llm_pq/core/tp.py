from .global_context import *
import torch.distributed as dist
def tp_launcher(ranks, backend='nccl'):
    dist.new_group(ranks, backend=backend)
    set_tp_group_ranks(ranks)