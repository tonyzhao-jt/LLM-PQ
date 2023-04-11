import torch
from operator import mul
from functools import reduce

def get_size_cuda(model_or_tensor, unit='bytes'):
    if isinstance(model_or_tensor, torch.Tensor):
        size = model_or_tensor.element_size() * reduce(mul, model_or_tensor.shape)
    elif isinstance(model_or_tensor, torch.nn.Module):
        params = list(model_or_tensor.parameters())
        total_params = sum(p.element_size() * reduce(mul, p.shape, 1) for p in params if p.is_cuda)
        total_buffers = sum(b.element_size() * reduce(mul, b.shape, 1) for b in model_or_tensor.buffers() if b.is_cuda)
        size = total_params + total_buffers
    else:
        raise TypeError("Input must be a PyTorch tensor or model")

    if unit == 'bytes':
        size_val = size
    elif unit == 'KB':
        size_val = size / 1024
    elif unit == 'MB':
        size_val = size / (1024 * 1024)
    elif unit == 'GB':
        size_val = size / (1024 * 1024 * 1024)
    else:
        raise ValueError(f"Invalid unit: {unit}")

    return size_val