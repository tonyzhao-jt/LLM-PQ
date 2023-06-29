import torch
import torch.nn as nn
from qllm.models import opt 

from bitsandbytes.nn.modules import Linear8bitLt
import bitsandbytes as bnb

import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock


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

class SQBnBLinear(nn.Module):
    def __init__(self, linear:nn.Linear, name, scales):
        super().__init__()
        weight = linear.weight.data.clone() / (scales)
        if linear.bias is not None:
            bias = linear.bias.data.clone() / (scales)
        # use SQ scales to scale the weight and bias
        linear_custom = Linear8bitLt(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            has_fp16_weights=False,
            threshold=0.0, # by default set 6.0, but is weight-only quantization.
        )
        linear_custom.state.force_no_igemmlt = False # this option is also important to determine whether bitsandbytes is used kernel.

        linear_custom.weight = bnb.nn.Int8Params(
            weight, requires_grad=False, has_fp16_weights=False
        ).to(linear.weight.dtype)
        linear_custom.bias.data = bias
        self.linear = linear_custom.cuda()
        self.scales = scales
    
    @torch.no_grad()
    def forward(self, input_):
        # scale input
        input_ = input_.mul_(self.scales)
        res = self.linear(input_) 
        return res 


if __name__ == '__main__':
    scale_file_name = 'opt-13b.pt'
    scale_file_name = torch.load(scale_file_name)
    # print(scale_file_name.keys()) # only activations
    opt_13b, tokenizer = opt.load_pretained_model_from_net('facebook/opt-13b')
    # device = torch.device('cuda')
    # test_li = nn.Linear(100, 100).to(device).half()
    # test_input = torch.randn(100, 100).to(device).half()
    # test_output = test_li(test_input)
    # print(test_output)
    # sqbn_li = SQBnBLinear(test_li, 'test', 1.0)
    # sqbn_output = sqbn_li(test_input)
    # sqbn_li.linear.to(device)
    # print(torch.allclose(test_output, sqbn_output, atol=1e-3))
