# set different portion of model to quantized to show the difference
# top mid bottom
# take a model, given the compression rate (reduce memory required)
# given different types of inidcator, test the accuracy of them in the gptq

from llm_pq.utils import (
    save_with_pickle, partition_a_into_b_bins
)

from qllm.models import opt
from qllm.models import bloom

# setup ilp configs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default='125m')
parser.add_argument('--model_name', type=str, default='opt')
parser.add_argument('--file_name', type=str, default=None)
# specify which part of model to be quantized
parser.add_argument('--part_idx', type=int, default=0)
args = parser.parse_args()

# model size
model_name = args.model_name # 'opt'
model_size = args.model_size # '66b'


if model_name == 'opt':
    assert model_size in opt.model_cards.keys(), f"model_size {model_size} not in opt.model_cards.keys()"
    config = opt.model_cards[model_size]
    num_layers = config.num_hidden_layers
elif model_name == 'bloom':
    assert model_size in bloom.model_cards.keys(), f"model_size {model_size} not in bloom.model_cards.keys()"
    config = bloom.model_cards[model_size]
    num_layers = config.n_layer

# fake T
T = [0, 1] * num_layers
L = num_layers
available_bits = [4, 16] # show the difference 
# partition the L into three slots
parts = partition_a_into_b_bins(L, 3)
# get the range of each part
range_of_parts = []
start_idx = 0
for part in parts:
    range_of_parts.append([start_idx, start_idx + part])
    start_idx += part
part_idx = args.part_idx
target_part = range_of_parts[part_idx]
bit_assignment = {}
for layer in range(L):
    if layer < target_part[1] and layer >= target_part[0]:
        bit_assignment[layer] = [4, 4]
    else:
        bit_assignment[layer] = [16, 16]

print(bit_assignment)

folder_name = 'bit_result'
file_name = f'{part_idx}_{model_name}_{model_size}_bit_ass.pkl'
save_with_pickle(bit_assignment, file_name, folder_name)
