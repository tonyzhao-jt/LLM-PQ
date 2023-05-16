import pickle
import torch
import numpy as np
from qpipe.utils import (
    save_with_pickle,
)

from utils import simple_model_info_parser
args = simple_model_info_parser()
model_size = args.model_size
model_name = args.model_name

folder_path = "/workspace/qpipe/3rd_party/gptq/zeroShot"
if model_name == 'bloom':
    model_pretrained_name = f"bigscience/bloom-{model_size}"
if model_name == 'opt':
    model_pretrained_name = f"facebook/opt-{model_size}"

model_pretrained_name = model_pretrained_name.replace('/', '_')
dataset = 'c4'
file_name = f"{model_pretrained_name}_{dataset}_stat.pkl"

# load the data
if model_name == 'opt':
    from qllm.models.OPT.opt import model_cards, get_available_models
    config = model_cards[model_size]
    num_layers = config.num_hidden_layers
if model_name == 'bloom':
    from qllm.models.BLOOM.bloom import model_cards, get_available_models
    config = model_cards[model_size]
    num_layers = config.n_layer

L = num_layers
abs_file_path = f"{folder_path}/{file_name}"

# load model
with open(abs_file_path, 'rb') as f:
    collected_information = pickle.load(f)

x_quant_method = "token"
w_quant_method = "tensor"

# the indicator is calculated by
# we only consider the attention and the ffn
available_bits = [2, 3, 4, 8, 16] # regard 8-bit as same
available_bits = list(set(available_bits))
BITs = [
    (i, j) for i in available_bits for j in available_bits
]

# get indicator for each layer
# final result should be a dict with respect to different layer and BITs
def calculate_indicator(bit, x_max, x_min, w_max, w_min, is_ffn=False):
    ind = 0
    x_max = x_max.astype(np.float64)
    x_min = x_min.astype(np.float64)
    w_max = w_max.astype(np.float64)
    w_min = w_min.astype(np.float64)
    # first item of the indicator
    # weight_always_tensor
    w_mag = (w_max - w_min).max()
    # weight always 
    item_num = 1
    if x_quant_method == 'tensor':
        item_num = x_max.shape[1]
        x_max = np.max(x_max.abs(), x_min.abs())
    # first term
    if item_num != 1:
        first_term = w_mag * x_max * item_num
    else:
        first_term = (w_mag * x_max).sum()
    # second item of the indicator
    w_max = np.abs(w_max).max()
    item_num = 1
    if x_quant_method == 'tensor':
        x_mag = np.abs((x_max - x_min)).max()
        item_num = x_max.shape[1]
    else:
        x_mag = np.abs((x_max - x_min))
    
    if item_num != 1:
        second_term = w_max * x_mag * item_num
    else:
        second_term = (w_max * x_mag).sum()

    slot = 2 ** (bit) - 1
    ind = (first_term / slot) ** 2 + (second_term / slot) ** 2


    if is_ffn:
        ind *= 2
    return ind

omega = np.zeros((L, len(BITs)))
for layer_idx, collected_stat in collected_information.items():
    # for bloom. qkv
    if model_name == 'bloom':
        qkv_quant_stat = collected_stat['self_attention.query_key_value']
        out_stat = collected_stat['self_attention.dense']
        mlp_first = collected_stat['mlp.dense_h_to_4h']
        mlp_second = collected_stat['mlp.dense_4h_to_h']
        # calculate the indicator
    elif model_name == 'opt':
        q_quant_stat = collected_stat['self_attn.q_proj']
        k_quant_stat = collected_stat['self_attn.k_proj']
        v_quant_stat = collected_stat['self_attn.v_proj']
        out_stat = collected_stat['self_attn.out_proj']
        mlp_first = collected_stat['fc1']
        mlp_second = collected_stat['fc2']
    
    for b_idx, bit in enumerate(BITs):
        bit_1, bit_2 = bit
        # calculate qkv
        if model_name == 'bloom':
            qkv_xmax, qkv_xmin = qkv_quant_stat['xmax'], qkv_quant_stat['xmin']
            qkv_wmax, qkv_wmin = qkv_quant_stat['wmax'], qkv_quant_stat['wmin']
            out_xmax, out_xmin = out_stat['xmax'], out_stat['xmin']

            qkv_ind = calculate_indicator(bit_1, qkv_xmax, qkv_xmin, qkv_wmax, qkv_wmin)
            qkv_out = calculate_indicator(bit_1, out_xmax, out_xmin, qkv_wmax, qkv_wmin)
            #
        elif model_name == 'opt':
            q_xmax, q_xmin = q_quant_stat['xmax'], q_quant_stat['xmin']
            q_wmax, q_wmin = q_quant_stat['wmax'], q_quant_stat['wmin']
            k_xmax, k_xmin = k_quant_stat['xmax'], k_quant_stat['xmin']
            k_wmax, k_wmin = k_quant_stat['wmax'], k_quant_stat['wmin']
            v_xmax, v_xmin = v_quant_stat['xmax'], v_quant_stat['xmin']
            v_wmax, v_wmin = v_quant_stat['wmax'], v_quant_stat['wmin']
            out_xmax, out_xmin = out_stat['xmax'], out_stat['xmin']
            if x_quant_method == 'tensor':
                q_xmax = q_xmax.max()
                q_xmin = q_xmin.min()
                k_xmax = k_xmax.max()
                k_xmin = k_xmin.min()
                v_xmax = v_xmax.max()
                v_xmin = v_xmin.min()
            
            qkv_ind = calculate_indicator(bit_1, q_xmax, q_xmin, q_wmax, q_wmin)
            qkv_ind += calculate_indicator(bit_1, k_xmax, k_xmin, k_wmax, k_wmin)
            qkv_ind += calculate_indicator(bit_1, v_xmax, v_xmin, v_wmax, v_wmin)
            qkv_out = calculate_indicator(bit_1, out_xmax, out_xmin, q_wmax, q_wmin)
        
        mlp_first_xmax, mlp_first_xmin = mlp_first['xmax'], mlp_first['xmin']
        mlp_first_wmax, mlp_first_wmin = mlp_first['wmax'], mlp_first['wmin']
        mlp_second_xmax, mlp_second_xmin = mlp_second['xmax'], mlp_second['xmin']
        mlp_second_wmax, mlp_second_wmin = mlp_second['wmax'], mlp_second['wmin']
        mlp_first_ind = calculate_indicator(bit_2, mlp_first_xmax, mlp_first_xmin, mlp_first_wmax, mlp_first_wmin, is_ffn=True)
        mlp_second_ind = calculate_indicator(bit_2, mlp_second_xmax, mlp_second_xmin, mlp_second_wmax, mlp_second_wmin, is_ffn=True)
        # calculate the indicator sum for the selection of layer
        ind_res = qkv_ind + qkv_out + mlp_first_ind + mlp_second_ind
        print("layer: {}, bit: {}, indicator: {}".format(layer_idx, bit, ind_res))
        omega[layer_idx, b_idx] = ind_res

# store t
folder_name = 'generated_ind'
file_name = f'gen_{model_name}_{model_size}_ind.pkl'
save_with_pickle(omega, file_name, folder_name)