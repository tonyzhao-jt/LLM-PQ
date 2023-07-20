import pickle
import torch
import numpy as np
from shaq.utils import (
    save_with_pickle, get_available_bits_pair
)

from utils import simple_model_info_parser, model_config_and_decoder_layers, get_available_candidate_bits
import copy 
def generate_indicator(model_name, model_size, folder_path, fast=True):
    if model_name == 'bloom':
        model_pretrained_name = f"bigscience/bloom-{model_size}"
    if model_name == 'opt':
        model_pretrained_name = f"facebook/opt-{model_size}"

    model_pretrained_name = model_pretrained_name.replace('/', '_')
    dataset = 'c4'
    file_name = f"{model_pretrained_name}_{dataset}_stat.pkl"

    # load the data
    config, num_layers = model_config_and_decoder_layers(model_name, model_size)
    # weight size
    hd_size = config.hidden_size


    L = num_layers
    abs_file_path = f"{folder_path}/{file_name}"

    # load model
    with open(abs_file_path, 'rb') as f:
        collected_information = pickle.load(f)

    dur = copy.deepcopy(collected_information['duration']) 
    del collected_information['duration']
    x_quant_method = "token"
    w_quant_method = "tensor"

    # the indicator is calculated by
    # we only consider the attention and the ffn
    available_bits = get_available_candidate_bits() # regard 8-bit as same
    BITs = get_available_bits_pair(available_bits)

    # get indicator for each layer
    # final result should be a dict with respect to different layer and BITs
    def calculate_indicator(bit, x_var, w_max, w_min, is_ffn=False):
        if bit in ['8:tc', '8:tc-li']:
            bit = 8
        ind = 0
        # get s_w^2 and d_w
        # focus on get s_w
        w_mag = (w_max - w_min).max()
        s_w = w_mag / (2 ** (bit - 1) - 1)
        # result = s_w^2 d_w varx
        ind = s_w ** 2 * x_var
        # check if ind is 0-d tensor
        ind = ind.sum()
        # since dw is same execpt ffn, so we only multiply ffn with 4
        if is_ffn:
            ind *= 4
        # weight_dim
        # w_size = hd_size ** 2
        # ind *= w_size
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
            # attention block
            # qkv + output
            if model_name == 'bloom':
                qkv_x_var = qkv_quant_stat['x_var']
                qkv_wmax, qkv_wmin = qkv_quant_stat['wmax'], qkv_quant_stat['wmin']

                out_x_var = out_stat['x_var']
                out_wmax, out_wmin = out_stat['wmax'], out_stat['wmin']
                qkv_ind = calculate_indicator(bit_1, qkv_x_var, qkv_wmax, qkv_wmin)
                qkv_out = calculate_indicator(bit_1, out_x_var, out_wmax, out_wmin)

            elif model_name == 'opt':
                q_x_var = q_quant_stat['x_var']
                k_x_var = k_quant_stat['x_var']
                v_x_var = v_quant_stat['x_var']

                q_wmax, q_wmin = q_quant_stat['wmax'], q_quant_stat['wmin']
                k_wmax, k_wmin = k_quant_stat['wmax'], k_quant_stat['wmin']
                v_wmax, v_wmin = v_quant_stat['wmax'], v_quant_stat['wmin']

                out_x_var = out_stat['x_var']
                out_wmax, out_wmin = out_stat['wmax'], out_stat['wmin']
                
                qkv_ind = calculate_indicator(bit_1, q_x_var, q_wmax, q_wmin)
                qkv_ind += calculate_indicator(bit_1, k_x_var, k_wmax, k_wmin)
                qkv_ind += calculate_indicator(bit_1, v_x_var, v_wmax, v_wmin)
                qkv_out = calculate_indicator(bit_1, out_x_var, out_wmax, out_wmin)
            
            # mlp block
            mlp_fc1_x_var = mlp_first['x_var']
            mlp_fc1_wmax, mlp_fc1_wmin = mlp_first['wmax'], mlp_first['wmin']
            mlp_fc2_x_var = mlp_second['x_var']
            mlp_fc2_wmax, mlp_fc2_wmin = mlp_second['wmax'], mlp_second['wmin']

            mlp_first_ind = calculate_indicator(bit_2, mlp_fc1_x_var, mlp_fc1_wmax, mlp_fc1_wmin, is_ffn=True)
            mlp_second_ind = calculate_indicator(bit_2, mlp_fc2_x_var, mlp_fc2_wmax, mlp_fc2_wmin, is_ffn=True)
            # calculate the indicator sum for the selection of layer
            ind_res = qkv_ind + qkv_out + mlp_first_ind + mlp_second_ind
            print("layer: {}, bit: {}, indicator: {}".format(layer_idx, bit, ind_res))
            omega[layer_idx, b_idx] = ind_res

    return omega, dur
args = simple_model_info_parser()
model_size = args.model_size
model_name = args.model_name
folder_path = "/workspace/qpipe/3rd_party/gptq/zeroShot"
omega, dur = generate_indicator(model_name, model_size, folder_path, fast=True)
# store t
folder_name = 'generated_ind'
file_name = f'gen_{model_name}_{model_size}_ind.pkl'
save_with_pickle(omega, file_name, folder_name)