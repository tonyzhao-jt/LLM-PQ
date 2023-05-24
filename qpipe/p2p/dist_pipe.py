from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch 
import torch.nn as nn 
import torch.distributed as dist
import queue
import threading
import time

from .comm import _send_tensor, _recv_tensor
from .TYPES import *
from . import util

import qpipe
from qpipe.utils import to_device


class ConditionQueue(queue.Queue):
    """A Queue with a public `condition: threading.Condition` variable for synchronization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.condition = threading.Condition()


class AbstractTensorExchangeThread(threading.Thread):
    """Abstract tensor exchange thread."""

    def __init__(self):
        super().__init__()
        self._pre_hooks = []
        self._post_hooks = []

    def register_pre_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register hook with signature: `hook(*args)`."""
        self._pre_hooks.append((hook, args))

    # Python 3.7 type hinting doesn't support the real hook function signature, which is more like:
    # `Callable[[Tuple[torch.Tensor], ...], None]`
    def register_post_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register hook with signature: `hook(tensors, *args)`."""
        self._post_hooks.append((hook, args))

    def run(self):
        """Still-abstract thread run method."""
        raise NotImplementedError

    def _call_pre_hooks(self):
        for hook, args in self._pre_hooks:
            hook(*args)

    def _call_post_hooks(self, tensors):
        for hook, args in self._post_hooks:
            hook(tensors, *args)

class SimpleQueueThread(AbstractTensorExchangeThread):
    def __init__(self, lock_queue, work_queue, dst_func):
        super().__init__()
        self._dst_func = dst_func
        self.lock_queue = lock_queue
        self.work_queue = work_queue
        # self.lock_queue = ConditionQueue(maxsize=1)
        self._evt_stop_thread = threading.Event()
    
    def stop(self) -> None:
        """Direct the thread to stop."""
        self._evt_stop_thread.set()
    
    def run(self):
        while self.lock_queue.empty():
            # print("watiing")
            time.sleep(0.001)
        while True:
            # get element from workqueue
            with self.work_queue.condition:
                while self.work_queue.empty():
                    self.work_queue.condition.wait()
                payload = self.work_queue.get(block=False)
                self.work_queue.condition.notify_all()
            # process element
            self._dst_func(payload)

class TensorSendThread(AbstractTensorExchangeThread):
    """Thread for sending tensors."""

    def __init__(self, queue_out: ConditionQueue, dst_rank: int):
        super().__init__()
        self._queue_out = queue_out
        self._dst_rank = dst_rank
        self._evt_stop_thread = threading.Event()
        self.on_device = torch.device("cuda", qpipe._globals.__DEVICE__INDEX__) if qpipe._globals.__DEVICE__INDEX__ is not None else torch.device('cpu')

        # buffer
        self.tensor_count = torch.tensor(-1, dtype=torch.int)

    def stop(self) -> None:
        """Direct the thread to stop."""
        with self._queue_out.condition:
            self._evt_stop_thread.set()
            self._queue_out.condition.notify_all()

    def run(self):
        """Dequeue tensors and send them."""
        while not self._evt_stop_thread.is_set():
            with self._queue_out.condition:
                while self._queue_out.empty():
                    if self._evt_stop_thread.is_set():
                        return
                    self._queue_out.condition.wait()
                payload = self._queue_out.get(block=False)
                self._queue_out.condition.notify_all()
            # Avoids pickling tensors if payload is a Tensor or Tuple[Tensor, ...]
            if isinstance(payload, tuple):
                objs = payload
                tensor_count = torch.tensor(len(objs), dtype=torch.int)
            else:
                objs = (payload,)
                tensor_count = self.tensor_count
            # pickle as needed
            tensors = ()
            tensor_sizes = ()
            for obj in objs:
                if isinstance(obj, torch.Tensor):
                    tensor, tensor_size = obj, torch.LongTensor([-1])
                else:
                    tensor, tensor_size = util.object_to_tensor(obj, None)
                tensors += (tensor,)
                tensor_sizes += (tensor_size,)
            dist.send(tensor=tensor_count, dst=self._dst_rank, tag=TAG_BASE_DATA+TAG_TENSOR_COUNT)
            # pre/post hooks should only wrap tensor send, not any pickling work (above)
            self._call_pre_hooks()
            for tensor, tensor_size in zip(tensors, tensor_sizes):
                dist.send(tensor=tensor_size, dst=self._dst_rank,
                          tag=TAG_BASE_DATA+TAG_TENSOR_PICKLED_SIZE)
                _send_tensor(tensor, self._dst_rank, TAG_BASE_DATA)
            self._call_post_hooks(tensors)


class TensorRecvThread(AbstractTensorExchangeThread):
    """Thread for receiving tensors."""

    def __init__(self, queue_in: ConditionQueue, src_rank: int):
        super().__init__()
        self._queue_in = queue_in
        self._src_rank = src_rank
        self._evt_stop_thread = threading.Event()
        self.on_device = torch.device("cuda", qpipe._globals.__DEVICE__INDEX__) if qpipe._globals.__DEVICE__INDEX__ is not None else torch.device('cpu')
        # buffer

    def stop(self) -> None:
        """Direct the thread to stop."""
        self._evt_stop_thread.set()

    def run(self):
        """Receive tensors and enqueue them."""
        while True:
            tensor_count = torch.tensor(0, dtype=torch.int)
            ircv_req = dist.irecv(tensor=tensor_count, src=self._src_rank,
                                  tag=TAG_BASE_DATA+TAG_TENSOR_COUNT)
            ircv_req_t = util.DistRequestWaitDaemon(ircv_req)
            ircv_req_t.start()
            while ircv_req_t.is_alive():
                if self._evt_stop_thread.is_set():
                    return
                # TODO: we're basically spinning...
                time.sleep(0.001)
            tensors = ()
            tensor_sizes = ()
            # pre/post hooks should only wrap tensor recv, not any unpickling work (further below)
            self._call_pre_hooks()
            for _ in range(abs(tensor_count)):
                tensor_size = torch.LongTensor([-1])
                dist.recv(tensor=tensor_size, src=self._src_rank,
                          tag=TAG_BASE_DATA+TAG_TENSOR_PICKLED_SIZE)
                tensor = _recv_tensor(self._src_rank, TAG_BASE_DATA)
                tensor_sizes += (tensor_size,)
                tensors += (tensor,)
            self._call_post_hooks(tensors)
            # unpickle as needed
            objs = ()
            for tensor, tensor_size in zip(tensors, tensor_sizes):
                obj = tensor if tensor_size < 0 else util.tensor_to_object(tensor, tensor_size)
                objs += (obj,)
            # if tensor_count >= 0, then the original payload was a tuple
            payload = objs if tensor_count >= 0 else objs[0]
            # Blocks if queue is full, which then blocks receiving more tensors (as intended)
            # Worker thread must be running to avoid indefinite blocking
            with self._queue_in.condition:
                while self._queue_in.full():
                    self._queue_in.condition.wait()
                self._queue_in.put(payload)
                self._queue_in.condition.notify_all()

class TensorWorkThread(threading.Thread):
    """Thread for processing tensors."""

    def __init__(self, queue_in: ConditionQueue, queue_out: ConditionQueue, callback: Callable):
        super().__init__()
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._callback = callback
        self._evt_stop_thread = threading.Event()
        self.stage_device = torch.device("cuda", qpipe._globals.__DEVICE__INDEX__)

        # add device.
        if isinstance(callback, nn.Module):
            self.on_device = self._callback.on_device
        else:
            self.on_device = None
        self.device_index = qpipe._globals.__DEVICE__INDEX__
        
        self.enable_tp = False
        self._tp_signal_threads = {}
        self._tp_cond_queue = {}
        self.global_rank = qpipe._globals.__GLOBAL__RANK__
    

    def stop(self) -> None:
        """Direct the thread to stop."""
        if self.enable_tp:
            for trd in self._tp_signal_threads.values():
                trd.stop()
                trd.join()

        with self._queue_in.condition:
            self._evt_stop_thread.set()
            self._queue_in.condition.notify_all()


    def run(self):
        """Dequeue, process, enqueue."""
        # Empty inbound queue before stopping
        while True:
            with self._queue_in.condition:
                while self._queue_in.empty():
                    if self._evt_stop_thread.is_set():
                        return
                    self._queue_in.condition.wait()
                tensor_in = self._queue_in.get(block=False)
                self._queue_in.condition.notify_all()
            
            if self.on_device is not None:
                tensor_in = to_device(tensor_in, self.on_device) # move to gpu if needed
                tensor_out = self._callback.decode(tensor_in)
                if self.device_index is None:
                    tensor_out = to_device(tensor_out, 'cpu') # move to cpu / test
            else:
                tensor_out = self._callback(tensor_in)

            if tensor_out is not None:
                # Sender thread must be running to avoid indefinite blocking
                with self._queue_out.condition:
                    while self._queue_out.full():
                        self._queue_out.condition.wait()
                    self._queue_out.put(tensor_out)
                    self._queue_out.condition.notify_all()

class DistP2pPipelineStage:
    """
    The singleton distributed P2P pipeline stage context manager.
    Creates receiver, sender, worker, and results processing threads when their respective
    optional parameters are specified.
    Threads communicate with each other through data queues, where the exact configuration depends
    on which threads are requested.
    Parameters must be specified appropriately on each rank to form a functionally correct pipeline.
    Because there is (at most) one receiver thread, only one rank may specify `results_cb` and
    that rank must not have a `work_cb` (be a stage) in the middle of the work pipeline.
    If it's the first work stage, that rank must also be the data source feeding `enqueue_tensor`
    (i.e., not receive inputs from a rank outside the work pipeline).
    If it's the last work stage, then `rank_dst` must be `None`, otherwise the results processing
    thread and sender thread would race for the data produced by the work thread.
    Otherwise, the rank specifying `results_cb` must not be in the work pipeline.
    Ranks that do nothing may specify `None` for all parameters.
    Parameters
    ----------
    rank_src : Optional[int]
        The rank to receive tensors from.
    rank_dst : Optional[int]
        The rank to send tensors to.
    work_cb : Optional[Callable]
        The worker callback - if None, received tensors are sent without modification.
    results_cb : Optional[Callable]
        The results callback.
    """

    def __init__(self, rank_src: Optional[int], rank_dst: Optional[int],
                 work_cb: Optional[Callable], results_cb: Optional[Callable[[Any], None]]):
        self._initialized = False
        self._queues = {}
        self._threads = {}
        self._create_stage(rank_src, rank_dst, work_cb, results_cb)

    def _create_stage(self, rank_src, rank_dst, work_cb, results_cb):
        self._queues['in'] = ConditionQueue(maxsize=1)
        self._queues['out'] = ConditionQueue(maxsize=1)
        self._queues['res'] = ConditionQueue(maxsize=1)

        # worker threads
        if work_cb is None:
            # Short-circuit from the inbound queue (can relay data without a worker thread)
            self._queues['out'] = self._queues['in']
        else:
            self._threads['work'] = TensorWorkThread(self._queues['in'], self._queues['out'],
                                                     work_cb)
        # return result to target
        if results_cb is not None:
            queue_res = self._queues['out'] if rank_dst is None else self._queues['res']
            self._threads['res'] = TensorWorkThread(queue_res, None, results_cb)

        if rank_dst is not None:
            self._threads['send'] = TensorSendThread(self._queues['out'], rank_dst)

        if rank_src is not None:
            queue_in = self._queues['in'] if results_cb is None else self._queues['res']
            self._threads['recv'] = TensorRecvThread(queue_in, rank_src)

    def init(self) -> None:
        """Initialize the distributed context and threads."""
        assert not self._initialized
        self._initialized = True
        for thr in self._threads.values():
            thr.start()

    def shutdown(self) -> None:
        """Shutdown threads and the distributed context."""
        assert self._initialized
        self._initialized = False
        for thr in self._threads.values():
            thr.stop()
            thr.join()

    def register_recv_pre_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register a pre hook for tensor receive with signature: `hook(*args)`."""
        thr = self._threads.get('recv')
        if thr is not None:
            thr.register_pre_hook(hook, args)

    def register_recv_post_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register a post hook for tensor receive with signature: `hook(tensors, *args)`."""
        thr = self._threads.get('recv')
        if thr is not None:
            thr.register_post_hook(hook, args)

    def register_send_pre_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register a pre hook for tensor send with signature: `hook(*args)`."""
        thr = self._threads.get('send')
        if thr is not None:
            thr.register_pre_hook(hook, args)

    def register_send_post_hook(self, hook: Callable[..., None], args: tuple) -> None:
        """Register a post hook for tensor send with signature: `hook(tensors, *args)`."""
        thr = self._threads.get('send')
        if thr is not None:
            thr.register_post_hook(hook, args)

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *args):
        self.shutdown()

    def enqueue_tensor(self, tensor: torch.Tensor) -> None:
        """Insert data into the pipeline."""
        assert self._initialized
        queue_in = self._queues['in']
        with queue_in.condition:
            while queue_in.full():
                queue_in.condition.wait()
            queue_in.put(tensor)
            queue_in.condition.notify_all()

def dist_p2p_pipeline_stage_factory(stage_ranks: List[int], data_rank: int, rank: int,
                                    stage: Optional[int], module,
                                    handle_results_cb: Callable[[Any], None]) \
    -> DistP2pPipelineStage:
    """Get a P2P pipeline stage instance."""
    if rank == data_rank:
        if stage is None:
            # We're data_rank w/out a module shard
            rank_src = stage_ranks[-1]
            rank_dst = stage_ranks[0]
            work_cb = None
        else:
            # We're simultaneously data_rank and a pipeline stage
            # In this case, the current p2p design requires that we must be the first stage
            if stage != 0:
                raise ValueError(f"Data rank must be stage=0 or stage=None, but stage={stage}")
            # Degenerate case when we're both data_rank and the only stage
            rank_src = stage_ranks[-1] if len(stage_ranks) > 1 else None
            rank_dst = stage_ranks[1] if len(stage_ranks) > 1 else None
            work_cb = module
        # While the handle_results_cb parameter isn't optional, we should assert it anyway.
        # If None, DistP2pPipelineStage would loop results back to its input queue, then the first
        # module shard would try to process the results tensors, which it would fail to unpack.
        # It wouldn't be obvious from the error that the real problem was handle_results_cb=None.
        assert handle_results_cb is not None
        results_cb = handle_results_cb
    elif stage is None:
        # We're completely idle
        rank_src = None
        rank_dst = None
        work_cb = None
        results_cb = None
    else:
        # We're not data_rank, but we have a module shard (possibly first and/or last stage)
        rank_src = data_rank if stage == 0 else stage_ranks[(stage - 1)]
        rank_dst = data_rank if stage == len(stage_ranks) - 1 else stage_ranks[(stage + 1)]
        work_cb = module
        results_cb = None

    print(f"rank_src: {rank_src}, rank_dst: {rank_dst}, rank: {rank}, results_cb: {results_cb}")
    return DistP2pPipelineStage(rank_src, rank_dst, work_cb, results_cb)