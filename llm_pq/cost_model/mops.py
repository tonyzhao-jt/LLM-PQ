def PROJ_MOPS(b, h1):
    return 2 * b * h1 + h1 ** 2

def BMM_MOPS(b, h1, i):
    return b * h1 + b * h1 * i + b * i 

def MLP_MOPS(b, h1, h2, i):
    return b * h1 + b * h2 + h1 * h2

def SELF_ATTN_MOPS_PARAMS(b, h1, i, bit):
    weight_size = 4 * h1 * h1 * 2 # fp16 bydefault
    qkv_act_size = 2 * 4 * b * 1 * h1 * 2 # fp16 bydefault
    bmm_act_size = 4 * b * h1 * i * 2 + 2 * b * i # fp16 bydefault
    kv_concat_size = 2 * b * i * h1 * 2 # fp16 bydefault
    layer_norm_size = b * i * h1 * 2 # fp16 bydefault
    dequant_size = 0
    bit = int(bit) if str(bit).isnumeric() else bit
    if bit == '8:tc':
        # it reduce both mops for weight and activation
        # but also reduce the concat/split cost for KV (int8)
        weight_size = weight_size / 2
        qkv_act_size = qkv_act_size / 2
        kv_concat_size = kv_concat_size / 2
        bmm_act_size = bmm_act_size / 2
        layer_norm_size = layer_norm_size / 2
    else:
        if bit == '8:tc-li':
            weight_size = weight_size / 2
            dequant_size = qkv_act_size
        elif bit == 8:
            weight_size = weight_size / 2
            dequant_size = weight_size
        elif bit == 4:
            weight_size = weight_size / 4
            dequant_size = weight_size
        elif bit == 2:
            weight_size = weight_size / 8
            dequant_size = weight_size

    return weight_size, qkv_act_size, kv_concat_size, bmm_act_size, layer_norm_size, dequant_size

def FFN_MOPS_PARAMS(b, h1, h2, bit):
    weight_size = 2 * h1 * h2 * 2 # fp16 bydefault
    act_size = b * (h1 + h2) * 2 # fp16 bydefault
    layer_norm_size = b * h1 * 2
    bit = int(bit) if str(bit).isnumeric() else bit
    dequant_size = 0
    if bit == '8:tc':
        # it reduce both mops for weight and activation
        weight_size = weight_size / 2
        act_size = act_size / 2
        layer_norm_size = layer_norm_size / 2
        dequant_size = 0
    else:
        if bit == '8:tc-li':
            weight_size = weight_size / 2
            dequant_size = act_size
        elif bit == 8:
            weight_size = weight_size / 2
            dequant_size = weight_size
        elif bit == 4:
            weight_size = weight_size / 4
            dequant_size = weight_size
        elif bit == 2:
            weight_size = weight_size / 8
            dequant_size = weight_size

    return weight_size, act_size, layer_norm_size, dequant_size