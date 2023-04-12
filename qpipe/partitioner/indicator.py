# indicator.py
# give each layer its sensitivity to quantization
import numpy as np 
def assign_uniform_indicator(L, available_bits):
    indicator = {}
    for l_idx in range(L):
        max_ind = np.random.uniform(0, 1)
        for bit in available_bits:
            indicator[(l_idx, bit)] = max_ind * 16 / bit # smaller, bigger
    return indicator