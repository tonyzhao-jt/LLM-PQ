import torch 
import torch.nn as nn

a = torch.rand(1,2,3, device='cuda:0')
b = torch.rand(1,2,3,device='cuda:0' )

@torch.cuda.synchronize()
def add(a, b):
    return a + b

c = add(a,b)