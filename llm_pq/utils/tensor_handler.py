import torch 
def object_to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, int):
        return torch.tensor(obj, dtype=torch.long)
    elif isinstance(obj, float):
        return torch.tensor(obj, dtype=torch.float)
    elif isinstance(obj, bool):
        return torch.tensor(obj, dtype=torch.bool)
    elif isinstance(obj, list):
        return [object_to_tensor(o) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(object_to_tensor(o) for o in obj)
    elif isinstance(obj, dict):
        return {k: object_to_tensor(v) for k, v in obj.items()}

