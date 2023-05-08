# use rpc to replace the communication used in p2p for nccl backend
import torch
from torch.distributed import rpc
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote

RPC_TAG_SEND_DATA = "111"

@rpc.functions.async_execution
def send_data(data):
    # get reference to receiver process
    receiver = rpc.remote("worker", receive_data)
    # send data to receiver and wait for result
    receiver.rpc_async(data).wait()

# define a remote function that receives data and sends result
@rpc.functions.async_execution
def receive_data(data):
    # perform computation and return result
    result = ...

    # get reference to sender process
    sender = rpc.remote("master", rpc.backend_registry.BackendType.TENSORPIPE)

    # send result to sender
    sender.rpc_async(result)

    return result