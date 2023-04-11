import torch 
import torch.nn as nn
from time import perf_counter

# run decoder gemm in cuda and cpu 
M, K, N = 512, 4096, 2048
sample_x = torch.randn(M, K)
linear = nn.Linear(K, N, bias=True)

cnt_times = 100

# cpu
start = perf_counter()
for i in range(cnt_times):
    y = linear(sample_x)
cpu_time = (perf_counter() - start) / cnt_times


# cuda
linear = linear.cuda()
sample_x = sample_x.cuda()
torch.cuda.synchronize()
start = perf_counter()
for i in range(cnt_times):
    y = linear(sample_x)
torch.cuda.synchronize()
cuda_time = (perf_counter() - start) / cnt_times

print(f'cpu time: {cpu_time}, cuda time: {cuda_time}')
