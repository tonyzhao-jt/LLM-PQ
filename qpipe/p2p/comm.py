import queue
import threading
from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
import torch.distributed as dist

import qpipe
from qpipe.logger import logger
from qpipe._globals import __PIPELINE__MODEL__PARALLEL__GROUP__

from .TYPES import *
sched_q = queue.Queue()
stop_event = threading.Event()
CMD_STOP = 0
CMD_SCHED = 1
def handle_cmd(cmd: int, tensors: Tuple[torch.Tensor, ...]) -> None:
    """Process received commands."""
    if cmd == CMD_STOP:
        logger.info("handle_cmd: stop")
        stop_event.set()
    elif cmd == CMD_SCHED:
        logger.info("handle_cmd: sched")
        sched_q.put(tuple(t.tolist() for t in tensors))
    else:
        logger.warning("handle_cmd: Unknown command: %s", cmd)


def _send_tensor(tensor, dst, tag_base, fn_send=dist.send):
    __DEVICE__INDEX__ = qpipe._globals.__DEVICE__INDEX__
    __PIPELINE__MODEL__PARALLEL__GROUP__ = qpipe._globals.__PIPELINE__MODEL__PARALLEL__GROUP__
    if __DEVICE__INDEX__ is not None:
        device = torch.device("cuda", __DEVICE__INDEX__)
    else:
        device = torch.device('cpu')
    # optimize by packing dtype and shape length into one message
    shape_len = len(tensor.shape)
    tensor_dtype_shapelen = torch.tensor([TORCH_TYPES_ENUM[tensor.dtype], shape_len], # [2] element. dtype, and how many tensors
                                         dtype=torch.int, device=device)
    results = []
    if __PIPELINE__MODEL__PARALLEL__GROUP__ is None:
        results.append(fn_send(tensor=tensor_dtype_shapelen, dst=dst,
                    tag=tag_base+TAG_TENSOR_DTYPE_SHAPELEN))
        if shape_len > 0:
            tensor_shape = torch.tensor(tensor.shape, dtype=torch.int, device=device)
            results.append(fn_send(tensor=tensor_shape, dst=dst, tag=tag_base+TAG_TENSOR_SHAPE))
        results.append(fn_send(tensor=tensor, dst=dst, tag=tag_base+TAG_TENSOR))
    else:
        # set group for fn_send
        results.append(fn_send(tensor=tensor_dtype_shapelen, dst=dst,
                    tag=tag_base+TAG_TENSOR_DTYPE_SHAPELEN, group=__PIPELINE__MODEL__PARALLEL__GROUP__))
        if shape_len > 0:
            tensor_shape = torch.tensor(tensor.shape, dtype=torch.int, device=device)
            results.append(fn_send(tensor=tensor_shape, dst=dst, tag=tag_base+TAG_TENSOR_SHAPE, group=__PIPELINE__MODEL__PARALLEL__GROUP__))
        results.append(fn_send(tensor=tensor, dst=dst, tag=tag_base+TAG_TENSOR, group=__PIPELINE__MODEL__PARALLEL__GROUP__))
    return results

def _recv_tensor(src, tag_base):
    __DEVICE__INDEX__ = qpipe._globals.__DEVICE__INDEX__
    __PIPELINE__MODEL__PARALLEL__GROUP__ = qpipe._globals.__PIPELINE__MODEL__PARALLEL__GROUP__
    if __DEVICE__INDEX__ is not None:
        device = torch.device("cuda", __DEVICE__INDEX__)
    else:
        device = torch.device('cpu')
    tensor_dtype_shapelen = torch.zeros(2, dtype=torch.int, device=device)
    if __PIPELINE__MODEL__PARALLEL__GROUP__ is None:
        dist.recv(tensor=tensor_dtype_shapelen, src=src, tag=tag_base+TAG_TENSOR_DTYPE_SHAPELEN)
        dtype = TORCH_TYPES[tensor_dtype_shapelen[0]]
        shape_len = tensor_dtype_shapelen[1]
        tensor_shape = torch.zeros(shape_len, dtype=torch.int, device=device)
        if shape_len > 0:
            dist.recv(tensor=tensor_shape, src=src, tag=tag_base+TAG_TENSOR_SHAPE)

        tensor = torch.tensor((), dtype=dtype, device=device).new_empty(tensor_shape.tolist())
        dist.recv(tensor=tensor, src=src, tag=tag_base+TAG_TENSOR)
    else:
        dist.recv(tensor=tensor_dtype_shapelen, src=src, tag=tag_base+TAG_TENSOR_DTYPE_SHAPELEN, group=__PIPELINE__MODEL__PARALLEL__GROUP__)
        dtype = TORCH_TYPES[tensor_dtype_shapelen[0]]
        shape_len = tensor_dtype_shapelen[1]
        tensor_shape = torch.zeros(shape_len, dtype=torch.int, device=device)
        if shape_len > 0:
            dist.recv(tensor=tensor_shape, src=src, tag=tag_base+TAG_TENSOR_SHAPE, group=__PIPELINE__MODEL__PARALLEL__GROUP__)

        tensor = torch.tensor((), dtype=dtype, device=device).new_empty(tensor_shape.tolist())
        dist.recv(tensor=tensor, src=src, tag=tag_base+TAG_TENSOR, group=__PIPELINE__MODEL__PARALLEL__GROUP__)

    return tensor


