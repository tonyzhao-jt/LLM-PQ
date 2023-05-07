
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
import torch.distributed as dist
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import qpipe
from qpipe.utils import to_device
from qpipe.rpc import get_neighbor_ranks
from qpipe.logger import logger

class DistRpcPipelineStage:
    """Wrap a module that is not RPC-aware to manage threading and memory."""
    # NOTE: message ordering is NOT enforced!

    def __init__(self, module_cls, module_args: Optional[tuple]=None,
                 module_kwargs: Optional[dict]=None, \
                 stage_id:int = None, comm_type:str=None,):
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
        # TODO: replace with single node strategy
        self.stage_device = torch.device("cuda", qpipe._globals.__DEVICE__INDEX__)
        self.stage_id = stage_id
        self.comm_type = comm_type

        # self._module = module_cls
        self._module = module_cls.to_here().local_value()
        self.enable_tp = False
        self.global_rank = qpipe._globals.__GLOBAL__RANK__
        if stage_id == -1:
            self.enable_tp = True
            self.tp_group = qpipe._globals.__TENSOR__MODEL__PARALLEL__GROUP__
        self.tp_rrefs = []

        self._next_rref = None
        self._results_to = None
        self._results_cb = None


        logger.info("Stage %d on %d", self.stage_id, self.global_rank)
    
    def set_tp_ref(self, cur_tp_rrefs) -> None:
        self.enable_tp = True if len(cur_tp_rrefs) > 0 else False
        if self.enable_tp:
            for ref in cur_tp_rrefs:
                self.tp_rrefs.append(ref)
            nums = len(self.tp_rrefs)
            self.tp_group = qpipe._globals.__TENSOR__MODEL__PARALLEL__GROUP__
            logger.info(f"rank {self.global_rank} setup tprefs: {nums}")

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
        
        if self.enable_tp:
            for ref in self.tp_rrefs:
                print("wait for ready", self.global_rank)
                ref.rpc_sync().wait_for_ready()
                ref.rpc_async().__call__(inputs)
        logger.info(f"on rank {self.global_rank} - stage: {self.stage_id}")
        inputs = to_device(inputs, self.stage_device)

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

    def __init__(self, stage_rrefs: List[rpc.RRef], tp_rrefs, results_to: Union[int, rpc.WorkerInfo, str],
                 results_cb: Callable[[Any], None]):
        super().__init__()
        self._rref_list = stage_rrefs
        self._tp_rrefs = tp_rrefs
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
        print("try link pipeline")
        futs = [self._rref_list[i].rpc_async().set_next(self._rref_list[i + 1])
                for i in range(n_stages - 1)]
        futs.append(self._rref_list[-1].rpc_async().set_results(results_to, results_cb))
        torch.futures.wait_all(futs)
        print("link pipeline done")
        futs = [self._rref_list[i].rpc_async().set_tp_ref(self._tp_rrefs[self._rref_list[i]])
                for i in range(n_stages)]
        torch.futures.wait_all(futs)
        print("TP set") 
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


def dist_rpc_pipeline_factory(head_stage_ranks, hard_device_mesh, all_ranks_involved, rref_dict, results_to: int,
                              results_cb: Callable[[Any], None]) -> DistRpcPipeline:
    """Get an RPC pipeline instance."""
    stage_rrefs = []
    tp_rrefs = {}

    num_stages = len(head_stage_ranks)
    # for each device, create a pipeline stage
    for rank in all_ranks_involved:
        module_rref = rref_dict[rank]
        if rank in head_stage_ranks:
            neighbor_ranks = get_neighbor_ranks(hard_device_mesh, rank)
            stage_id = head_stage_ranks.index(rank)
            next_rank = (rank + 1) % num_stages
            if next_rank in neighbor_ranks:
                comm_type = f"cuda:{neighbor_ranks.index(next_rank)}"
            else:
                comm_type = "cpu"
            
            rref = rpc.remote(f"worker{rank}", _dist_rpc_pipeline_stage_factory, args=([module_rref]),
                            kwargs={'stage_id': stage_id, 'module_kwargs': {}, \
                                    'comm_type': comm_type})
            stage_rrefs.append(rref)
            tp_rrefs[rref] = []
        else:
            stage_id = -1
            comm_type = 'cpu'
            tp_rref = rpc.remote(f"worker{rank}", _dist_rpc_pipeline_stage_factory, args=([module_rref]),
                            kwargs={'stage_id': stage_id, 'module_kwargs': {}, \
                                    'comm_type': comm_type})
            if rref not in tp_rrefs:
                tp_rrefs[rref] = [tp_rref]
            else:
                tp_rrefs[rref].append(tp_rref)
    
    assert len(stage_rrefs) == len(head_stage_ranks), \
        f"stage_rrefs length ({len(stage_rrefs)}) doesn't match head_stage_ranks length " 

    logger.info("results to rank %d", results_to)
    logger.info("results callback %s", results_cb.__name__ if results_cb else "None")
    logger.info("======= Pipeline created =======")
    logger.info("with %d stages", len(stage_rrefs))
    return DistRpcPipeline(stage_rrefs, tp_rrefs, results_to, results_cb)