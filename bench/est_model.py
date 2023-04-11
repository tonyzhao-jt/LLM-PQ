import argparse
import os
import pandas as pd 
from qllm.utils import ModelMemEstimator
def convert_to_unit(data, unit):
    unit = unit.upper()
    if unit == 'MB':
        return data / 1024 / 1024
    elif unit == 'KB':
        return data / 1024
    elif unit == 'B':
        return data
    elif unit == 'GB':
        return data / 1024 / 1024 / 1024
    else:
        raise ValueError('unit should be one of MB, KB, B')
    
# set logger level to debug
import logging
logging.basicConfig(level=logging.DEBUG)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter parameters for the model.')
    parser.add_argument('--h2', type=int, default=512, help='First dimension of fc2')
    parser.add_argument('--h1', type=int, default=256, help='Model hidden space')
    parser.add_argument('--b', type=int, default=32, help='Token batch size')
    parser.add_argument('--s', type=int, default=64, help='Input sentence length')
    parser.add_argument('--n', type=int, default=128, help='Number of generated tokens')
    parser.add_argument('--l', type=int, default=1, help='Number of layers')
    parser.add_argument('--unit', type=str, default='MB', help='Unit of the output (MB, GB, TB)')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--store', action='store_true', help='Store the result in a csv file (results.csv)')
    parser.add_argument('--bits', type=int, default=16, help='bit')

    args = parser.parse_args()
    
    h1 = args.h1
    h2 = args.h2
    b = args.b
    s = args.s
    n = args.n
    l = args.l
    bit = args.bits

    unit = args.unit
    
    print(f'h2: {h2}')
    print(f'h1: {h1}')
    print(f'b: {b}')
    print(f's: {s}')
    print(f'n: {n}')
    print(f'l: {l}')
    print(f'units: {unit}')

    model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n)
    
    hidden_state = convert_to_unit(model_mem_estimator.estimate_hidden_space(), unit)
    kv_mem = convert_to_unit(l * model_mem_estimator.estimate_single_layer_kv_cache(), unit)
    weight_mem = convert_to_unit(model_mem_estimator.calculate_multiple_layer_decoder_layer_mem(l), unit)
    print(model_mem_estimator.calculate_single_decoder_layer_mem())
    # correct value with bits
    hidden_state = hidden_state * bit / 32
    kv_mem = kv_mem * bit / 32
    weight_mem = weight_mem * bit / 32
    print("Estimated Hidden Space size", hidden_state)
    print(f"Estimated KV for {l} layer", kv_mem)
    print(f"Estimated memory for {l} decoder layer", weight_mem)
    
    # open an csv to store the result, 
    # create csv to store the result
    if args.store:
        if not os.path.exists('results.csv'):
            df = pd.DataFrame(columns=['model', 'h2', 'h1', 'b', 's', 'n', 'l', 'hidden_state', 'kv_mem', 'weight_mem'])
            df.to_csv('results.csv', index=False)
            
        df = pd.read_csv('results.csv')
        df = df._append({'model': args.model, 'h2': h2, 'h1': h1, 'b': b, 's': s, 'n': n, 'l': l, 'hidden_state': hidden_state, 'kv_mem': kv_mem, 'weight_mem': weight_mem}, ignore_index=True)
        df.to_csv('results.csv', index=False)

