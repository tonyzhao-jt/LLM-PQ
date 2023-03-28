import torch 
import torch.nn as nn 
from torch.distributed import rpc
# the module should be wrapped with a queue
# imitate the implementation of: https://pytorch.org/docs/stable/_modules/torch/distributed/pipeline/sync/pipe.html#Pipe
def _setup_module_devices(partitioned_module):
    for node_name, modulde_to_device_plan in partitioned_module.items():
        # for the master node, launch modules in the same process
        if node_name == 'master':
            for device, module in modulde_to_device_plan.items():
                module = module.to(device)
        else:
            # use rpc remote
            for device, module in modulde_to_device_plan.items():
                module = rpc.remote(node_name, module.to, args=(device,))

class QPipe(nn.Module):
    def __init__(
        self,
        module_partition_plan: dict,
        chunks: int = 1,
    ) -> None:
        # Check if RPC framework is initialized.
        if not torch.distributed.rpc._is_current_rpc_agent_set():
            raise RuntimeError(
                'Please initialize RPC framework for Pipe using '
                'torch.distributed.rpc.init_rpc')
        chunks = int(chunks)
        if chunks <= 0:
            raise ValueError("number of chunks must be positive integer")

        self.chunks = chunks
        # different from the implementation of torch
        # since we need to use the inter-node communication, we directly takes the partition plan as input
        # partition_rule = {
        #     'master': {0: block_1},
        #     'worker1': {0: block_2, 1: block_3},
        # }
        _setup_module_devices(module_partition_plan)
qpipe = QPipe()