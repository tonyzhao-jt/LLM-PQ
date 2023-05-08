import torch 
import torch.distributed as dist

def create_device_mesh(rank, local_rank, world_size):
    backend = dist.get_backend()
    node_first_rank = rank - local_rank
    # get first_rank of each node and ngpus for each node
    # sort by first_rank, then we got the whole device mesh
    # dist.init_process_group(backend='gloo', init_method='env://') # no need to initialize

    if backend == 'gloo':
        node_info = torch.tensor([node_first_rank, local_rank], dtype=torch.int64)
        node_info_list = [torch.zeros(len(node_info), dtype=torch.int64) for _ in range(world_size)]
    else:
        device = torch.device('cuda:{}'.format(local_rank))
        node_info = torch.tensor([node_first_rank, local_rank], dtype=torch.int32, device=device)
        node_info_list = [torch.zeros(len(node_info), dtype=torch.int32, device=device) for _ in range(world_size)]
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
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

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