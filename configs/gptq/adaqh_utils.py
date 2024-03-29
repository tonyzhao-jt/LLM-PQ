import torch.nn as nn 
import torch 
import argparse
import pickle
import os 

# add new space to argparser
def add_argparser(parser: argparse.ArgumentParser):
    # add the ada file_path
    parser.add_argument('--ada-file', type=str, default=None)
    # motivation and debug
    parser.add_argument('--mixed-bit', action="store_true")
    return parser

mixed_precision_result = []
def mixed_bit_handler(args, bit_for_layer):
    if type(bit_for_layer) == list or type(bit_for_layer) == tuple:
        bit_for_layer = bit_for_layer[0] # for the moment we didn't decompose MLP and ATTN
    if not args.mixed_bit:
        return bit_for_layer
    else:
        if bit_for_layer == 4:
            # randomly choose from 4 and 8
            return 4 if torch.rand(1) < 0.5 else 8
        elif bit_for_layer == 3:
            # randomly choose from 3 and 4
            return 3 if torch.rand(1) < 0.5 else 4
        

# handle the precision that is not well tackled by gptq
available_non_gptq = ['bitsandbytes']
def handle_non_gptq_impl(layer:nn.Module, impl_type:str="bitsandbytes"):
    os.environ['PERF_MODE'] = "0" # not in perf mode
    # os.environ['LP_BITS_THRESHOLD'] = "0.3"

    if impl_type not in available_non_gptq:
        raise NotImplementedError(f"impl_type {impl_type} not implemented")
    if impl_type == 'bitsandbytes':
        # create bitsandbytes layer
        from lptorch import quantize_linear_module_with_bit
        os.environ['Q_METHOD'] = "BITSANDBYTES"
        quantize_linear_module_with_bit(layer, bit=8)
        os.environ['Q_METHOD'] = "ADALINEAR"
        print(f"quantize layer with bitsandbytes")
        # import pdb; pdb.set_trace()
        # print(layer)

custom_precisions = [8]
def customize_precision(layer:nn.Module, bit=8):
    if bit in custom_precisions:
        handle_non_gptq_impl(layer, impl_type='bitsandbytes')
        return True # pass the current layer
    else:
        return False 

# handle the llm int8
def quant_llm_int8(layer, threshold):
    os.environ['PERF_MODE'] = "0" # not in perf mode
    original_threshold = os.environ.get('LP_BITS_THRESHOLD', "6.0") # store current
    os.environ['LP_BITS_THRESHOLD'] = threshold
    from lptorch import quantize_linear_module_with_bit
    os.environ['Q_METHOD'] = "BITSANDBYTES"
    quantize_linear_module_with_bit(layer, bit=8)
    os.environ['Q_METHOD'] = "ADALINEAR"
    os.environ['LP_BITS_THRESHOLD'] = original_threshold # recover
    return layer

threshold_dict = {}
def customize_llm_int8(layer:nn.Module, idx=0, lower_threshold=False):
    if threshold_dict.get(idx, None) is None:
        threshold_dict[idx] = float(os.environ.get('LP_BITS_THRESHOLD', 6.0)) # default provided by HF
    else:
        if lower_threshold:
            print("Nan detected, lower the threshold for layer {}".format(idx))
            threshold_dict[idx] = threshold_dict[idx] * 0.9
    thresh_ = threshold_dict[idx]
    # print(threshold_dict)
    return quant_llm_int8(layer, str(thresh_))

import pickle
def log_customize_info(model_name:str):
    if len(threshold_dict) > 0:
        print("customize llm int8 threshold:")
        for k, v in threshold_dict.items():
            print(f"layer {k} threshold: {v}")
        with open(f"customize_info_{model_name.replace('/', '_')}.pkl", 'wb') as f:
            pickle.dump(threshold_dict, f)

def read_ada_file(file_path:str, layers:list):
    # or file not exists
    if file_path is not None and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            bit_assignment = pickle.load(f)
        # check numbers
        assert len(bit_assignment) == len(layers), "bit assignment length is not equal to layer length"
    else:
        bit_assignment = None 
    return bit_assignment