# based on the config to generate the random assignments
import random
import pickle
from llm_pq.utils import (
    save_with_pickle,
)

from utils import simple_model_info_parser
args = simple_model_info_parser()
model_size = args.model_size
model_name = args.model_name

if model_name == 'opt':
    from qllm.models.OPT.opt import model_cards, get_available_models
if model_name == 'bloom':
    from qllm.models.BLOOM.bloom import model_cards, get_available_models
available_models = get_available_models()
assert model_size in available_models, f'{model_size} not in available models {available_models}'
config = model_cards[model_size]
# layer of decoders
num_layers = config.num_hidden_layers
available_bits = [2, 3, 4, 8, 16]
bit_assignment = {}
for layer in range(num_layers):
    bit_assignment[layer] = [random.choice(available_bits) for i in range(2)] # qkv and FFN
print(bit_assignment)

folder_name = 'bit_for_gptq_test'
file_name = f'rand_{model_name}_{model_size}_bit_ass.pkl'
save_with_pickle(bit_assignment, file_name, folder_name)
