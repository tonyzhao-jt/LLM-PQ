import torch
import torch.nn as nn
import os
import torch.distributed as dist
from torch.distributed import rpc

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
model = nn.Linear(100, 100).to(device)

def check_mem_occupy():
    print(f"rank={rank}, local_rank={local_rank}, device={device}, memory_allocated={torch.cuda.memory_allocated(device)}, memory_cached={torch.cuda.memory_cached(device)}")

# check certain device memory occupation
def check_device_n_mem(p_device):
    print(f"device={p_device}, memory_allocated={torch.cuda.memory_allocated(p_device)}, memory_cached={torch.cuda.memory_cached(p_device)}")

def get_rref():
    return rpc.RRef(model)

__CURRENT__MODEL__ = None

def assign_model(rref):
    global __CURRENT__MODEL__
    __CURRENT__MODEL__ = rref.to_here().local_value()

def log_current_model():
    print("current model:", __CURRENT__MODEL__)
# init rpc
rpc.init_rpc(
    name=f"worker{rank}",
    rank=rank,
    world_size=WORLD_SIZE,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        init_method="env://",
    )
)

print(f"RPC initialized, rank={rank}, local_rank={local_rank}, world_size={WORLD_SIZE}")

if rank == 0:
    # get rref
    rref = get_rref()
    # get rref of other workers
    rref_dict = {}
    rref_list = []
    for i in range(1, WORLD_SIZE):
        rref_list.append(rpc.remote(f"worker{i}", get_rref))
        rref_dict[i] = rref_list[-1]
    check_mem_occupy()
    # call rpc
    for rref in rref_list:
        rref.to_here()
    check_mem_occupy()
    # create new linear to check mem
    print("create new linear")
    new_model = nn.Linear(100, 100).to(device)
    check_mem_occupy()
    rpc.remote(f"worker{1}", assign_model, args=(rref_dict[1],))
    # log
    rpc.remote(f"worker{1}", log_current_model)
    rpc.remote(f"worker{1}", check_mem_occupy)
else:
    # check cuda memory occupation
    check_mem_occupy()

# clean up rpc
rpc.shutdown()