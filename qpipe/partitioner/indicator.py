# indicator.py
# give each layer its sensitivity to quantization
import numpy as np 
def assign_uniform_indicator(L, available_bits):
    indicator = {}
    for l_idx in range(L):
        max_ind = np.random.uniform(0, 1)
        for b_idx, bit in enumerate(available_bits):
            if type(bit) is str:
                indicator[(l_idx, b_idx)] = max_ind * 16 / 8
            else:
                indicator[(l_idx, b_idx)] = max_ind * 16 / bit # smaller, bigger
    return indicator

# takes BITs pair as inputs 
def assign_omega_uniform(L, BITs):
    omega = np.zeros((L, len(BITs)))
    # suppose to collect from the model
    omega_dict = {
        2: 6.4,
        4: 1.6,
        8: 0.4,
        16: 0.1,
        '8:tc': 0.4,
        '8:tc-li': 0.4,
    }
    for l_idx in range(omega.shape[0]):
        for b_idx, bit_pair in enumerate(BITs):
            self_attn, ffn = bit_pair
            attn_omega = omega_dict[self_attn] * np.random.uniform(0, 1)
            ffn_omega = omega_dict[ffn] * np.random.uniform(0, 1)
            omega_layer_bitpair = attn_omega + ffn_omega
            omega[l_idx, b_idx] = omega_layer_bitpair
    return omega

# just fit in the memory
def assign_omega_constant(L, BITs):
    omega = np.zeros((L, len(BITs)))
    omega_dict = {
        2: 1,
        4: 1,
        8: 1,
        16: 1,
        '8:tc': 1,
        '8:tc-li': 1,
    }
    for l_idx in range(omega.shape[0]):
        for b_idx, bit_pair in enumerate(BITs):
            self_attn, ffn = bit_pair
            attn_omega = omega_dict[self_attn] * np.random.uniform(0, 1)
            ffn_omega = omega_dict[ffn] * np.random.uniform(0, 1)
            omega_layer_bitpair = attn_omega + ffn_omega
            omega[l_idx, b_idx] = omega_layer_bitpair
    return omega