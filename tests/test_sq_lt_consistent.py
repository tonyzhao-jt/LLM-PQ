# try approximate sq result using lt
from smoothquant.fake_quant import W8A8Linear as SQW8A8Linear
import torch
from torch import nn
from functools import partial

def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()
    return w, scales


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_()
    return t, scales

import bitsandbytes as bnb
import bitsandbytes.functional as bnb_func

class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', quantize_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x, s_x = self.act_quant(x)
        # print(self.act_quant, q_x, s_x)
        # print(q_x, self.weight)
        y = bnb_func.igemm(q_x.to(torch.int8), self.weight.to(torch.int8).t())
        y = y * s_x * self.s_w + self.bias
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight.detach().clone(), n_bits=8)  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight, new_module.s_w = quantize_weight_per_tensor_absmax(
                module.weight.detach().clone(), n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'


torch.manual_seed(0)

test_linear = nn.Linear(100, 100).cuda()
fake_input = torch.rand(100, 100).cuda()
fake_linear = SQW8A8Linear.from_float(test_linear, weight_quant='per_tensor', act_quant='per_tensor')
my_linear = W8A8Linear.from_float(test_linear, weight_quant='per_tensor', act_quant='per_tensor')

fake_out = fake_linear(fake_input)
my_fake_out = my_linear(fake_input)

print(torch.max(torch.abs(fake_out - my_fake_out)))