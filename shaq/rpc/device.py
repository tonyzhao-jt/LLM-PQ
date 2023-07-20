import torch 
import torch.distributed as dist
import numpy as np
import os 
from typing import Optional, Dict, Union, Callable, Any
DeviceType = Union[int, str, torch.device]
from shaq.logger import logger
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
    host = MASTER_ADDR = os.environ['MASTER_ADDR']
    port = MASTER_PORT = int(os.environ['MASTER_PORT'])
    backend = 'nccl'
    init_method = f'tcp://[{host}]:{port}'
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    logger.info("INIT NCCL WITH ADDRESS %s:%d" % (MASTER_ADDR, MASTER_PORT))

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


def build_device_maps(world_size: int, n_proc_per_node: int, rank: Optional[int] = None) -> Dict[str, Dict[DeviceType, DeviceType]]:
    is_master = rank is None
    device_maps: Dict[str, Dict[DeviceType, DeviceType]] = {}
    if is_master:
        for i in range(world_size):
            worker_local_rank = i % n_proc_per_node
            device_maps[f'worker{i}'] = {'cpu': worker_local_rank}
    else:
        local_rank = rank % n_proc_per_node
        for i in range(world_size):
            if i != rank:
                worker_local_rank = i % n_proc_per_node
                device_maps[f'worker{i}'] = {local_rank: worker_local_rank}
        device_maps['master'] = {local_rank: 'cpu'}
    return device_maps

def set_device_map(cur_rank, device_mesh, hard_device_mesh, rpc_options):
    # check whether hard_device mesh is in the right format
    for first_rank, ranks in hard_device_mesh.items():
        assert first_rank == ranks[0], "hard_device_mesh should be in the format of {first_rank: [all_ranks on the same node]}"
    stages_2d = np.array(list(device_mesh.values())).T
    # tp_group_size, stage_nums = stages_2d.shape
    # # create device_map
    # device_maps = {}
    # # start from each row
    # for j in range(stage_nums):
    #     # logger.info(f"stage {j} COMM Initialization")
    #     # iterate columns
    #     ranks_within_stage = stages_2d[:, j]
    #     next_stage = (j + 1) % stage_nums
    #     for i, rank in enumerate(ranks_within_stage):
    #         rank = int(rank)
    #         if rank != cur_rank:
    #             continue
    #         local_rank = get_local_rank_by_device_mesh(hard_device_mesh, rank)
    #         neighbor_ranks = get_neighbor_ranks(hard_device_mesh, rank)
    #         # for the time being, the communication between gpus on the same stage is done by p2p groups
    #         # so we don't need to set device map for them
    #         next_rank = int(stages_2d[i, next_stage])
    #         next_rank_local_rank = get_local_rank_by_device_mesh(hard_device_mesh, next_rank)
            # TODO: support the device setup later. 
            # set map
            # device_maps[f"worker{rank}"] = {local_rank: next_rank_local_rank}
            # rpc_options.set_device_map(f"worker{next_rank}", {local_rank: next_rank_local_rank})
            # rpc_options.set_devices([local_rank])
            # logger.info(f"set device map: worker{rank}-{local_rank}: to worker{next_rank}-{next_rank_local_rank}")
    return stages_2d, rpc_options