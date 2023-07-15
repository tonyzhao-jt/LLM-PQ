# test weight convert
'''
   convert weight name
   - for the moment, only opt is required to do such conversion
'''
# generation reference for the bloom and opt
from qllm.utils.argparser import model_config_argparser
from qllm.models import qllm_load_pretrained_from_size, bare_load_pretrained_from_size, create_empty_model
from qllm.utils import batch_encode_plus
import torch 

cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/data/llms/')
target_storage_folder = f'{cache_dir}/converted_weights'
# check if the folder exits, create one if not
import os
if not os.path.exists(target_storage_folder):
    os.makedirs(target_storage_folder)

args = model_config_argparser()
model_name = args.model_name
model_size = args.model_size
bare_model, tokenizer, key = bare_load_pretrained_from_size(model_name, model_size)
state_dict = bare_model.state_dict()
# only opt requires conversion for the moment
if model_name == 'opt':
    keys = list(state_dict.keys())
    for k in keys:
        # rules to change the state dict
        if 'fc1' in k:
            # replace .fc1 with .mlp.fc1
            new_k = k.replace('.fc1', '.mlp.fc1')
        # same to fc2
        elif 'fc2' in k:
            new_k = k.replace('.fc2', '.mlp.fc2')
        elif 'final_layer_norm' in k:
            if 'decoder.final_layer_norm' in k:
                continue # didn't change it
            new_k = k.replace('.final_layer_norm', '.mlp.final_layer_norm')
        else:
            continue
        # update
        state_dict[new_k] = state_dict.pop(k)
qllm_empty_model = create_empty_model(model_name, model_size)
qllm_empty_model.load_state_dict(state_dict) # test whether error
# save pretrained
try:
    path = os.path.join(target_storage_folder, f"{model_name}_{model_size}")
    qllm_empty_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print("new weight saved")
except:
    print("saving failed")