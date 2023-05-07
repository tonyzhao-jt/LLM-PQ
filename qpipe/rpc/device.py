import torch 
import torch.distributed as dist
import logging

from qpipe.logger import logger
# dist / comm utils
def create_device_mesh(rank, local_rank, world_size):
    node_first_rank = rank - local_rank
    # get first_rank of each node and ngpus for each node
    # sort by first_rank, then we got the whole device mesh
    dist.init_process_group(backend='gloo', init_method='env://')

    node_info = torch.tensor([node_first_rank, local_rank], dtype=torch.int64)
    node_info_list = [torch.zeros(len(node_info), dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(node_info_list, node_info)
    # dist.destroy_process_group()
    # print("Process group closed")
    # based on the first node, create a mesh with ranks has the same first rank on the row
    # and ranks has the same local rank on the column
    device_mesh = {}
    for i in range(world_size):
        first_rank, local_rank = node_info_list[i].tolist()
        if first_rank not in device_mesh:
            device_mesh[first_rank] = []
        device_mesh[first_rank].append(local_rank + first_rank)
    return device_mesh

def create_device_mesh_nccl(rank, local_rank, world_size):
    node_first_rank = rank - local_rank
    # get first_rank of each node and ngpus for each node
    # sort by first_rank, then we got the whole device mesh
    dist.init_process_group(backend='nccl', init_method='env://')

    device = torch.device('cuda', local_rank)
    node_info = torch.tensor([node_first_rank, local_rank], dtype=torch.int64, device = device)
    node_info_list = [torch.zeros(len(node_info), dtype=torch.int64, device = device) for _ in range(world_size)]
    dist.all_gather(node_info_list, node_info)
    # dist.destroy_process_group()
    # print("Process group closed")
    # based on the first node, create a mesh with ranks has the same first rank on the row
    # and ranks has the same local rank on the column
    device_mesh = {}
    for i in range(world_size):
        first_rank, local_rank = node_info_list[i].tolist()
        if first_rank not in device_mesh:
            device_mesh[first_rank] = []
        device_mesh[first_rank].append(local_rank + first_rank)
    return device_mesh


def get_neighbor_ranks(device_mesh, rank):
    for first_rank, ranks in device_mesh.items():
        if rank in ranks:
            return ranks
        
def get_local_rank_by_device_mesh(device_mesh, rank):
    for first_rank, ranks in device_mesh.items():
        if rank in ranks:
            return rank - first_rank

def set_device_map(rank, device_mesh, schedules, rpc_options):
    stage_rank_order = list(schedules.keys())
    if rank not in stage_rank_order:
        return # only when rank is used
    i = stage_rank_order.index(rank)
    # determine the mapping from stage to device
    cur_stage_rank = stage_rank_order[i]
    next_stage_rank = stage_rank_order[(i + 1) % len(stage_rank_order)] # possible to be in the same node
    # cur_stage local rank
    cur_stage_local_rank = get_local_rank_by_device_mesh(device_mesh, cur_stage_rank)
    # next_stage local rank
    next_stage_local_rank = get_local_rank_by_device_mesh(device_mesh, next_stage_rank)
    if cur_stage_rank == next_stage_rank:
        pass
    else:
        logger.info(f"set device map for worker{cur_stage_rank}:{cur_stage_local_rank} to worker{next_stage_rank}:{next_stage_local_rank}")
        rpc_options.set_device_map(f"worker{next_stage_rank}", {cur_stage_local_rank: next_stage_local_rank})