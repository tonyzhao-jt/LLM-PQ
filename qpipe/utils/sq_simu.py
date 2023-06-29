import torch
import torch.nn as nn
from qllm.models import opt 

from bitsandbytes.nn.modules import Linear8bitLt
import bitsandbytes as bnb

import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock

from lptorch import quantize_linear_module_with_bit
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.float()
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)
    
    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + '.self_attention.query_key_value']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)


import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model

import os 
def smooth_13b_model(opt_13b):

    scale_file_name = '/workspace/qpipe/scripts/accuracy/SQ_reproduce/opt-13b.pt'
    scales = torch.load(scale_file_name)
    # smooth the models
    model = opt_13b.float()
    smooth_lm(model, scales, alpha=0.5)
    # quant using the bitsandbytes
    # os.environ['Q_METHOD'] = 'BITSANDBYTES'
    # print(model)
    model_smoothquant_w8a8 = quantize_model(model)
    return model_smoothquant_w8a8.to(torch.float16)

# if __name__ == '__main__':
#     # load model
#     opt_13b, tokenizer = opt.load_pretained_model_from_net('facebook/opt-13b',dtype=torch.float32)
    # load act scales
    
    # print(scale_file_name.keys()) # only activations
    # device = torch.device('cuda')
    # test_li = nn.Linear(100, 100).to(device).half()
    # test_input = torch.randn(100, 100).to(device).half()
    # test_output = test_li(test_input)
    # print(test_output)
    # sqbn_li = SQBnBLinear(test_li, 'test', 1.0)
    # sqbn_output = sqbn_li(test_input)
    # sqbn_li.linear.to(device)
    # print(torch.allclose(test_output, sqbn_output, atol=1e-3))
