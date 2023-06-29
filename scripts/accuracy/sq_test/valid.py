from lptorch import construct_quantized_linear
import torch 
import torch.nn as nn 
sample_linear = nn.Linear(10, 10).cuda().half()
sample_input = torch.randn(10, 10).cuda().half()
b_linear = construct_quantized_linear(sample_linear, 8, constructor='bitsandbytes', sample_input=sample_input)
b_linear = b_linear.cuda()
b_out = b_linear(sample_input)
original_out = sample_linear(sample_input)
print(torch.allclose(b_out, original_out, atol=1e-3))

