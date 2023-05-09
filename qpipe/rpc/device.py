import torch 
import torch.distributed as dist
import numpy as np

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
            # TODO: support the device setup later. 
            # set map
            # device_maps[f"worker{rank}"] = {local_rank: next_rank_local_rank}
            rpc_options.set_device_map(f"worker{next_rank}", {local_rank: next_rank_local_rank})
            # rpc_options.set_devices([local_rank])
            logger.info(f"set device map: worker{rank}-{local_rank}: to worker{next_rank}-{next_rank_local_rank}")
    return stages_2d, rpc_options