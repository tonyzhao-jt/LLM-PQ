# hessian methods.
import pickle
import torch
import numpy as np
from qpipe.utils import (
    save_with_pickle, get_available_bits_pair
)

from utils import simple_model_info_parser, model_config_and_decoder_layers, get_available_candidate_bits
import copy 
import os

def generate_indicator_hess(model_name, model_size, folder_path):
    if model_name == 'bloom':
        model_pretrained_name = f"bigscience/bloom-{model_size}"
    if model_name == 'opt':
        model_pretrained_name = f"facebook/opt-{model_size}"

    model_pretrained_name = model_pretrained_name.replace('/', '_')
    dataset = 'c4'
    available_bits = get_available_candidate_bits() # regard 8-bit as same
    all_collected_data = {}
    for bit in available_bits:
        if type(bit) == str: 
            bit = 8 # 8:tc or 8:tc-li
        if bit == 16:
            continue
        file_name = f"{model_pretrained_name}_{dataset}_hess_stat_{bit}.pkl"
        # check if exists
        abs_file_path = f"{folder_path}/{file_name}"
        if not os.path.exists(abs_file_path):
            raise ValueError(f"file {abs_file_path} does not exist")
        # load data
        with open(abs_file_path, 'rb') as f:
            collected_information = pickle.load(f)
        all_collected_data[bit] = collected_information
    # check the data
    config, num_layers = model_config_and_decoder_layers(model_name, model_size)
    L = num_layers
    BITs = get_available_bits_pair(available_bits)

    dur_sum = 0
    for bit, data in all_collected_data.items():
        dur_sum += data['duration']
        del data['duration']

    # get indicator for each layer
    # final result should be a dict with respect to different layer and BITs
    def calculate_indicator(bit_pair, layer_idx, all_collected_data):
        atten_bit, ffn_bit = bit_pair
        if atten_bit in ['8:tc', '8:tc-li']:
            atten_bit = 8
        if ffn_bit in ['8:tc', '8:tc-li']:
            ffn_bit = 8
        if model_name == 'bloom':
            qkv_stat_name = 'self_attention.query_key_value'
            out_stat_name = 'self_attention.dense'
            qkv_names = [qkv_stat_name, out_stat_name]
            mlp_first_name = 'mlp.dense_h_to_4h'
            mlp_second_name = 'mlp.dense_4h_to_h'
            mlp_names = [mlp_first_name, mlp_second_name]
        elif model_name == 'opt':
            q_stat_name = 'self_attn.q_proj'
            k_stat_name = 'self_attn.k_proj'
            v_stat_name = 'self_attn.v_proj'
            out_stat_name = 'self_attn.out_proj'
            qkv_names = [q_stat_name, k_stat_name, v_stat_name, out_stat_name]
            mlp_first_name = 'fc1'
            mlp_second_name = 'fc2'
            mlp_names = [mlp_first_name, mlp_second_name]
        
        # def calculate_hessian_error(err, hess):
        #     eigvals = torch.linalg.eigvalsh(hess)
        #     top_eigval = eigvals[-1]
        #     err_h = err * top_eigval
        #     return err_h
        # get the corresponding data we need
        # self attn
        all_err_h = 0
        for qkv_name in qkv_names:
            if atten_bit == 16:
                continue
            try:
                err_h = all_collected_data[atten_bit][(layer_idx, qkv_name)]
            except:
                import pdb; pdb.set_trace()
            # err_h = calculate_hessian_error(err, hess)
            all_err_h += err_h
        for mlp_name in mlp_names:
            if ffn_bit == 16:
                continue
            err_h = all_collected_data[ffn_bit][(layer_idx, mlp_name)]
            # err_h = calculate_hessian_error(err, hess)
            all_err_h += err_h
        return all_err_h
    
    omega = np.zeros((L, len(BITs)))
    for i in range(L):
        for j, bit_pair in enumerate(BITs):
            omega[i, j] = calculate_indicator(bit_pair, i, all_collected_data)
    
    return omega, dur_sum

args = simple_model_info_parser()
model_size = args.model_size
model_name = args.model_name
folder_path = "/workspace/qpipe/3rd_party/gptq/zeroShot"
omega, dur = generate_indicator_hess(model_name, model_size, folder_path)
# store t
folder_name = 'generated_ind'
file_name = f'gen_{model_name}_{model_size}_hess_ind.pkl'
save_with_pickle(omega, file_name, folder_name)
