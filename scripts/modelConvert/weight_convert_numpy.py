# test weight convert
'''
   convert weight name
   - for the moment, only opt is required to do such conversion
'''
import os
import numpy as np
# generation reference for the bloom and opt
from qllm.utils.argparser import model_config_argparser
from qllm.models import bare_load_pretrained_from_size

from tqdm import tqdm

cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/data/llms/')
target_storage_folder = f'{cache_dir}/converted_weights_np'
# check if the folder exits, create one if not
if not os.path.exists(target_storage_folder):
    os.makedirs(target_storage_folder)

args = model_config_argparser()
model_name = args.model_name
model_size = args.model_size
bare_model, tokenizer, key = bare_load_pretrained_from_size(model_name, model_size)
state_dict = bare_model.state_dict()
# the folder path
path = os.path.join(target_storage_folder, f"{model_name}_{model_size}")
# mkdir if path not exists
if not os.path.exists(path):
    os.makedirs(path)
# only opt requires conversion for the moment
if model_name == 'opt':
    keys = list(state_dict.keys())
    for k in tqdm(keys):
        # rules to change the state dict
        if 'fc1' in k:
            # replace .fc1 with .mlp.fc1
            new_k = k.replace('.fc1', '.mlp.fc1')
        # same to fc2
        elif 'fc2' in k:
            new_k = k.replace('.fc2', '.mlp.fc2')
        elif 'final_layer_norm' in k:
            if 'decoder.final_layer_norm' in k:
                # didn't change it
                new_k = k
            else:
                new_k = k.replace('.final_layer_norm', '.mlp.final_layer_norm')
        else:
            new_k = k
        param_path = os.path.join(path, new_k)
        with open(param_path, 'wb') as f:
            # save the weight
            np.save(f, state_dict[k].numpy())
    print("Converted weights for opt done for model size {}".format(model_size))
else:
    # for bloom we don't need to change name
    keys = list(state_dict.keys())
    for k in tqdm(keys):
        param_path = os.path.join(path, k)
        with open(param_path, 'wb') as f:
            # save the weight
            np.save(f, state_dict[k].numpy())
    print("Converted weights for bloom done for model size {}".format(model_size))
