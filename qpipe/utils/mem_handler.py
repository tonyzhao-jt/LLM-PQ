import torch
from operator import mul
from functools import reduce

def get_size_cuda(model_or_tensor, unit='bytes'):
    if isinstance(model_or_tensor, torch.Tensor):
        if model_or_tensor.is_cuda:
            size = model_or_tensor.element_size() * reduce(mul, model_or_tensor.shape)
        else:
            size = 0
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


def get_size_cpu(model_or_tensor, unit='bytes'):
    if isinstance(model_or_tensor, torch.Tensor):
        if not model_or_tensor.is_cuda:
            size = model_or_tensor.element_size() * reduce(mul, model_or_tensor.shape)
        else:
            size = 0
    elif isinstance(model_or_tensor, torch.nn.Module):
        params = list(model_or_tensor.parameters())
        total_params = sum(p.element_size() * reduce(mul, p.shape, 1) for p in params if not p.is_cuda)
        total_buffers = sum(b.element_size() * reduce(mul, b.shape, 1) for b in model_or_tensor.buffers() if not b.is_cuda)
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


def get_iter_variable_size(obj, unit='B'):
    def get_iter_variable_inner(obj):
        total_size = 0
        if isinstance(obj, (tuple, list)):
            for item in obj:
                total_size += get_iter_variable_inner(item)
        elif isinstance(obj, dict):
            for _, item in obj.items():
                total_size += get_iter_variable_inner(item)
        elif isinstance(obj, torch.Tensor):
            tensor = obj
            total_size += tensor.element_size() * tensor.nelement()
        return total_size
    total_size = get_iter_variable_inner(obj)
    if unit == 'B':
        return total_size
    elif unit == 'KB':
        return total_size / 1024
    elif unit == 'MB':
        return total_size / (1024 * 1024)
    elif unit == 'GB':
        return total_size / (1024 * 1024 * 1024)
    else:
        raise ValueError(f"Invalid unit '{unit}', choose from 'B', 'KB', or 'MB'.")
