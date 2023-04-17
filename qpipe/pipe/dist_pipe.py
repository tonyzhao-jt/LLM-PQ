
# model factory
# We did three steps here
# 1. based on the partition rule, get the correct layers
# 2. put them to correct device
# 3. set the correct quantization level, load the correct weight (PS: Do later)
"""RPC communication module."""
import torch
import torch.nn as nn 
import threading
from torch.distributed import rpc
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from qpipe.utils import to_device
from qpipe.rpc import get_neighbor_ranks
from qpipe.logger import logger
class DistRpcPipelineStage:
    """Wrap a module that is not RPC-aware to manage threading and memory."""
    # NOTE: message ordering is NOT enforced!

    def __init__(self, module_cls: Type[nn.Module], module_args: Optional[tuple]=None,
                 module_kwargs: Optional[dict]=None, \
                 stage_id:int = None, local_rank:int=None, comm_type:str=None,):
        super().__init__()
        if module_args is None:
            module_args = ()
        if module_kwargs is None:
            module_kwargs = {}
        # _sem_fwd limits RPC threads in forward(), and thus data memory requirements (in + out).
        # _sem_mod limits Module thread parallelism, and thus processing memory requirements.
        # If each stage is configured for single-thread Module processing, then N=1.
        # Ideally, for _sem_mod value=N:
        # (1) N inputs are being or have been received (prior to forward() or waiting on _sem_mod)
        # (2) N inputs are processing (acquired _sem_mod)
        # (3) N outputs are sending or waiting to send (released _sem_mod)
        # More generally, however, the local stage may be backed up at any of these three steps,
        # depending on its performance relative to other stages and network conditions.
        self._sem_fwd = threading.Semaphore(value=3) # value = 3*N
        self._sem_mod = threading.Semaphore(value=1) # value = N
        # here should me a nn.Sequential Module, and a function that set quantization bit should be here
        # self._module = module_cls(*module_args, **module_kwargs)
        # TODO: replace with single node strategy
        shard_config = module_kwargs['shard_config']
        infer_configs = module_kwargs['infer_configs']
        module_cls._shard_model_current(shard_config)
        print(f"Stage {stage_id} module sharded")
        self.is_master = stage_id==0
        self._module = module_cls
        self._next_rref = None
        self._results_to = None
        self._results_cb = None

        self.stage_id = stage_id
        self.comm_type = comm_type
        self.stage_device = torch.device(f"cuda:{local_rank}")
        self.module_to(self.stage_device)
        bs, prompt_length, num_tokens_to_generate, request_numbers = infer_configs
        for request_id in range(request_numbers):
            self._module.init_kv_cache(bs, prompt_length, num_tokens_to_generate, request_id)
        print(f"Stage {stage_id} kv initialized")
    def module_to(self, *args, **kwargs) -> None:
        """Wrap the module's `nn.Module.to` method (`device` can be be a `str`)."""
        if self.is_master:
            self._module = self._module.to(*args, **kwargs)
        else:
            self._module.decoder_layers_to_device(*args, **kwargs)
        # print(self._module, args, kwargs)

    def set_next(self, stage_rref: rpc.RRef) -> None:
        """Set the RRef of the next pipeline stage - used by all stages except the last."""
        self._next_rref = stage_rref

    def set_results(self, results_to: Union[int, rpc.WorkerInfo, str],
                    results_cb: Callable[[Any], None]) -> None:
        """Set the results destination - used by only the last stage."""
        self._results_to = results_to
        self._results_cb = results_cb
        # print("set results", results_to, results_cb)

    def wait_for_ready(self) -> None:
        """Wait for this stage to be ready to receive data - MUST be called from previous stage."""
        # NOTE: This approach breaks down if the previous stage fails to send data afterward.
        self._sem_fwd.acquire() # pylint: disable=consider-using-with
        
    def __call__(self, inputs: Any) -> None:
        """Wrap the module's callable method."""
        inputs = to_device(inputs, self.stage_device)
        # print("on", self.stage_id)
        with self._sem_mod:
            outputs = self._module.decode(inputs)
        if self._next_rref is not None:
            # Sending must be asynchronous, otherwise we lose pipeline parallelism.
            # However, don't try to send until the next stage is ready.
            # If we were to initiate the async send (and then release _sem_fwd) too soon,
            # outbound data could get backlogged in this stage when the next stage is slow.
            self._next_rref.rpc_sync().wait_for_ready()
            self._next_rref.rpc_async().__call__(outputs)
        else:
            assert self._results_to is not None
            assert self._results_cb is not None
            # There's no synchronization with the results handler, just send the data.
            rpc.rpc_sync(self._results_to, self._results_cb, args=(outputs,))
        # Now release so that another microbatch may be received.
        self._sem_fwd.release()
    
    def test_fwd(self, input):
        # print("run", input)
        input = to_device(input, self.stage_device)
        outputs = self._module.decode(input)
        if self._next_rref:
            self._next_rref.rpc_async().test_fwd(outputs)
        else:
            assert self._results_to is not None
            assert self._results_cb is not None
            # There's no synchronization with the results handler, just send the data.
            rpc.rpc_sync(self._results_to, self._results_cb, args=(outputs,))

    def mod_register_buffer(self, *args, **kwargs) -> None:
        """Wrap the module's `register_buffer()` method."""
        return self._module.register_buffer(*args, **kwargs)

    def mod_register_forward_hook(self, *args, **kwargs) -> None:
        """Wrap the module's `register_forward_hook()` method."""
        return self._module.register_forward_hook(*args, **kwargs)

    def mod_register_forward_pre_hook(self, *args, **kwargs) -> None:
        """Wrap the module's `register_forward_pre_hook()` method."""
        return self._module.register_forward_pre_hook(*args, **kwargs)

class DistRpcPipeline:
    """A distributed RPC pipeline which links `DistRpcPipelineStage` RRefs."""

    def __init__(self, stage_rrefs: List[rpc.RRef], results_to: Union[int, rpc.WorkerInfo, str],
                 results_cb: Callable[[Any], None]):
        super().__init__()
        self._rref_list = stage_rrefs
        self._link_pipeline(results_to, results_cb)

    def rpc_register_buffer(self, name: str, tensors: List[Optional[torch.Tensor]],
                            **kwargs: dict) -> None:
        """Add buffers to RPC modules."""
        if len(tensors) != len(self._rref_list):
            raise ValueError(f"tensors length ({len(tensors)}) doesn't match pipeline length "
                             f"({len(self._rref_list)})")
        futs = [rref.rpc_async().mod_register_buffer(name, tensor, **kwargs)
                for rref, tensor in zip(self._rref_list, tensors)]
        torch.futures.wait_all(futs)

    # rpc_async returns a future object
    def rpc_register_forward_pre_hook(self, hook: Callable[..., None], first: bool=True) -> None:
        """Register forward pre hook."""
        rrefs = self._rref_list if first else self._rref_list[1:]
        hook_futures = [rref.rpc_async().mod_register_forward_pre_hook(hook) for rref in rrefs]
        torch.futures.wait_all(hook_futures)

    def rpc_register_forward_hook(self, hook: Callable[..., None], last: bool=True) -> None:
        """Register forward hook."""
        rrefs = self._rref_list if last else self._rref_list[:-1]
        hook_futures = [rref.rpc_async().mod_register_forward_hook(hook) for rref in rrefs]
        torch.futures.wait_all(hook_futures)

    def _link_pipeline(self, results_to, results_cb):
        n_stages = len(self._rref_list)
        futs = [self._rref_list[i].rpc_async().set_next(self._rref_list[i + 1])
                for i in range(n_stages - 1)]
        futs.append(self._rref_list[-1].rpc_async().set_results(results_to, results_cb))
        torch.futures.wait_all(futs)

        # [self._rref_list[i].rpc_sync().set_next(self._rref_list[i + 1])
        #         for i in range(n_stages - 1)]
        # self._rref_list[-1].rpc_sync().set_results(results_to, results_cb)

    def enqueue_tensor(self, tensor: torch.Tensor) -> None:
        """Insert data into the front of the pipeline."""
        self._rref_list[0].rpc_sync().wait_for_ready()
        self._rref_list[0].rpc_async().__call__(tensor)
        # self._rref_list[0].rpc_async().test_fwd(tensor.cpu()) # very important that the data sent should on the same device.

def _dist_rpc_pipeline_stage_factory(*args, **kwargs) -> DistRpcPipelineStage:
    """Get a `rpc.DistRpcPipelineStage` instance on the globally-configured `devices.DEVICE`."""
    stage = DistRpcPipelineStage(*args, **kwargs)
    return stage


def dist_rpc_pipeline_factory(model_cpu: nn.Module, sharding_strategy: dict, device_mesh, infer_configs, results_to: int,
                              results_cb: Callable[[Any], None]) -> DistRpcPipeline:
    """Get an RPC pipeline instance."""
    stage_rrefs = []
    stage_cnt = 0

    dst_rank_order = list(sharding_strategy.keys())
    stage_numbers = len(dst_rank_order)

    for dst_rank, stage_cfgs in sharding_strategy.items():
        neighbor_ranks = get_neighbor_ranks(device_mesh, dst_rank)
        if stage_cnt == stage_numbers - 1:
            next_rank = 0 # meet the last one
            comm_type = "cpu"
        else:
            next_rank = dst_rank_order[stage_cnt + 1]
            if next_rank in neighbor_ranks:
                comm_type = f"cuda:{neighbor_ranks.index(next_rank)}"
            else:
                comm_type = "cpu"
        dst_local_rank = neighbor_ranks.index(dst_rank)
        logger.info(f"Stage {stage_cnt} on {dst_rank} with {comm_type} communication to {next_rank}, local rank {dst_local_rank}")

        rref = rpc.remote(dst_rank, _dist_rpc_pipeline_stage_factory, args=(model_cpu,),
                           kwargs={'stage_id': stage_cnt, 'module_kwargs': {'infer_configs':infer_configs, 'shard_config': sharding_strategy[dst_rank]}, \
                                   'local_rank':dst_local_rank, 'comm_type': comm_type})
        rpc.remote(dst_rank, logger.info,
                    args=("======= Stage %d on %d =======", stage_cnt, dst_rank))
        stage_cnt += 1
        stage_rrefs.append(rref)

    logger.info("results to rank %d", results_to)
    logger.info("results callback %s", results_cb.__name__ if results_cb else "None")
    logger.info("======= Pipeline created =======")
    logger.info("with %d stages", len(stage_rrefs))
    return DistRpcPipeline(stage_rrefs, results_to, results_cb)