# add new space to argparser
import argparse
def add_argparser(parser: argparse.ArgumentParser):
    # add the ada file_path
    parser.add_argument('--ada-file', type=str, default=None)
    # motivation and debug
    parser.add_argument('--rand-bit', action="store_true")

    return parser

# handle the precision that is not well tackled by gptq
import torch.nn as nn 
import torch 
available_non_gptq = ['bitsandbytes']
def handle_non_gptq_impl(layer:nn.Module, impl_type:str="bitsandbytes"):
    os.environ['PERF_MODE'] = "0" # not in perf mode
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

import pickle
import os 
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