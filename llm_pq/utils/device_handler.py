# utils
import torch
def to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, tuple):
        return tuple(to_device(t, device) for t in tensor)
    elif isinstance(tensor, list):
        return [to_device(t, device) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: to_device(v, device) for k, v in tensor.items()}
    else:
        return tensor