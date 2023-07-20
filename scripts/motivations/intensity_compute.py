# compute intensity
# prefill intensity
def compute_proj_prefill(b, s, h1):
    flops = 2 * b * s * h1 ** 2
    mops = 2 * b * s * h1 + h1 ** 2
    return flops, mops 

def compute_BMM_prefill(b, s, h1):
    flops = 2 * b * s ** 2 * h1
    mops = 2 * b * s * h1 + b * s ** 2
    return flops, mops

def compute_MLP_prefill(b, s, h1, h2):
    flops = 2 * b * s * h1 * h2
    mops = b * s * h1  + b * s * h2 +  h1 * h2
    return flops, mops


# decode intensity
def compute_proj_decode(b, h1):
    flops = 2 * b * h1 ** 2
    mops = 2 * b * h1 + h1 ** 2
    return flops, mops

def compute_BMM_decode(b, t, h1):
    flops = 2 * b * t * h1 
    mops = b * h1 + b * h1 * t + b * t
    return flops, mops

def compute_MLP_decode(b, h1, h2):
    flops = 2 * b * h1 * h2
    mops = b * h1  + b * h2 +  h1 * h2
    return flops, mops


def calculate_intensity_layer_prefill(b, s, h1, h2):
    overall_flops = 0
    overall_mops = 0
    # 4proj 2BMM 2MLP
    flops, mops = compute_proj_prefill(b, s, h1)
    overall_flops += flops * 4 
    overall_mops += mops * 4
    flops, mops = compute_BMM_prefill(b, s, h1)
    overall_flops += flops * 2
    overall_mops += mops * 2
    flops, mops = compute_MLP_prefill(b, s, h1, h2)
    overall_flops += flops * 2
    overall_mops += mops * 2
    print("intensity of prefill layer: ", overall_flops / overall_mops)
    return overall_flops, overall_mops

def calculate_intensity_layer_decode(b, t, h1, h2):
    overall_flops = 0
    overall_mops = 0
    # 4proj 2BMM 2MLP
    flops, mops = compute_proj_decode(b, h1)
    overall_flops += flops * 4
    overall_mops += mops * 4
    flops, mops = compute_BMM_decode(b, t, h1)
    overall_flops += flops * 2
    overall_mops += mops * 2
    flops, mops = compute_MLP_decode(b, h1, h2)
    overall_flops += flops * 2
    overall_mops += mops * 2
    # add KV load
    overall_mops += b * h1 * t
    print("intensity of decode layer: ", overall_flops / overall_mops)
    return overall_flops, overall_mops

if __name__ == '__main__':
    b = 32 
    s = 512
    t = 512
    # opt 175
    h1 = 12288
    h2 = 49152
    calculate_intensity_layer_prefill(b, s, h1, h2)
    calculate_intensity_layer_decode(b, t, h1, h2)
    # opt 30b
    h1 = 7168
    h2 = 28672
    calculate_intensity_layer_prefill(b, s, h1, h2)
    calculate_intensity_layer_decode(b, t, h1, h2)
