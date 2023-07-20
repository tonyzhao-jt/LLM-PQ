'''
    Script modified from pipeEdge: https://github.com/usc-isi/PipeEdge/blob/a6f58254ee4f4faf524a85ea2c1090a6d92bf0a2/src/pipeedge/comm/p2p/__init__.py#L41
'''

import os 
import torch
import torch.distributed as dist
import threading
import time
from typing import Any, Callable, List, Optional, Tuple, Type, Union
from . import util
from .env import init_env, new_nccl_group
from .comm import handle_cmd, stop_event
from .device import create_device_mesh

from .TYPES import *
from .comm import _send_tensor, _recv_tensor


import shaq
# RPC CONTEXT
DistCmdHandler: Type = Callable[[int, Tuple[torch.Tensor, ...]], None]

class DistContext:
    """Parent class for distributed context managers."""

    def __init__(self, init_args: tuple, init_kwargs: dict):
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._world_size = init_kwargs['world_size']
        self._rank = init_kwargs['rank']
        self._initialized = False

    def init(self) -> None:
        """Initialize the distributed context."""
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the distributed context."""
        self._initialized = False

    def __enter__(self):
        assert not self._initialized
        self.init()
        return self

    def __exit__(self, *args):
        assert self._initialized
        self.shutdown()


class CommandThread(threading.Thread):
    """Thread for receiving commands."""

    def __init__(self, callback: DistCmdHandler):
        super().__init__()
        self._callback = callback
        self._evt_stop_thread = threading.Event()
        self.on_device = torch.device("cuda", shaq._globals.__DEVICE__INDEX__) if shaq._globals.__DEVICE__INDEX__ is not None else torch.device('cpu')
        self.src_rank = shaq._globals.__ROW__FIRST__RANK__
        # buffer
        self.tensor_cmd = torch.zeros(2, dtype=torch.int, device=self.on_device)

    def stop(self) -> None:
        """Direct the thread to stop."""
        self._evt_stop_thread.set()

    def run(self):
        """Listen for commands."""
        while True:
            # contains (1) CMD enumeration and (2) an optional tensor count
            tensor_cmd = torch.zeros(2, dtype=torch.int, device=self.on_device)
            ircv_req = dist.irecv(tensor=tensor_cmd, tag=TAG_BASE_CMD)
            ircv_req_t = util.DistRequestWaitDaemon(ircv_req)
            ircv_req_t.start()
            while ircv_req_t.is_alive():
                if self._evt_stop_thread.is_set():
                    return
                # TODO: we're basically spinning...
                time.sleep(0.1)
            cmd = int(tensor_cmd[0])
            _tensor_count = int(tensor_cmd[1])
            tensors = ()
            for _ in range(_tensor_count):
                # it would be nice if we could restrict src to the prior request's src, but the
                # ircv_req "distributed request object" API doesn't document a src property to use
                tensor = _recv_tensor(None, TAG_BASE_CMD)
                tensors += (tensor,)
            self._callback(cmd, tensors)

class DistP2pContext(DistContext):
    """
    The singleton distributed P2P context manager.
    Parameters
    ----------
    ipg_args : tuple
        Arguments for ``torch.distributed.init_process_group()``.
    ipg_kwargs : dict
        Keyword arguments for ``torch.distributed.init_process_group()``.
    cmd_cb : DistCmdHandler
        Command handler callback.
    """

    def __init__(self, ipg_args: tuple, ipg_kwargs: dict, cmd_cb: DistCmdHandler):
        super().__init__(ipg_args, ipg_kwargs)
        self._thread_cmd = CommandThread(cmd_cb)

        self.on_device = torch.device("cuda", shaq._globals.__DEVICE__INDEX__) if shaq._globals.__DEVICE__INDEX__ is not None else torch.device('cpu')

    def init(self) -> None:
        """Initialize the distributed context and threads."""
        super().init()
        # check whether we're already initialized
        if not dist.is_initialized():
            dist.init_process_group(*self._init_args, **self._init_kwargs)
        self._thread_cmd.start()

    def shutdown(self) -> None:
        """Shutdown threads and the distributed context."""
        super().shutdown()
        self._thread_cmd.stop()
        self._thread_cmd.join()
        dist.destroy_process_group()

    def cmd_broadcast(self, cmd: int, tensors: Optional[Tuple[torch.Tensor, ...]]=None) -> None:
        """Broadcast a command."""
        assert self._initialized
        if tensors is None:
            tensors = ()
        tensor_cmd = torch.tensor([cmd, len(tensors)], dtype=torch.int, device=self.on_device)
        reqs = []
        for dst in range(self._world_size):
            if dst != self._rank:
                reqs.append(dist.isend(tensor_cmd, dst=dst, tag=TAG_BASE_CMD))
                for tensor in tensors:
                    reqs += _send_tensor(tensor, dst, TAG_BASE_CMD, fn_send=dist.isend)
        for req in reqs:
            req.wait()



