'''
    Sharded the weight based on the partition
'''
# example utils
from example_utils import create_uniform_sharding_strategies, parse_args
from qllm.models import qllm_load_pretrained_from_size
import pickle
import os 
import torch 
# inputs: sharding strategies
# outputs: 
#   sharded decoder weights sharded_decode_weight_1.bin, etc.
#   non-decoder weights non_decoder_weight_1.bin, etc.

args = parse_args()
bitwidth = args.bitwidth
if type(bitwidth) is not int and bitwidth.isnumeric():
    args.bitwidth = int(bitwidth)

# test case
model_name = args.model_name
model_size = args.model_size
'''
    Performance mode
    - when you only concerns the performance rather than accuracy
'''

# load weight
target_storage_folder = '/data/llms/converted_weights'
# load model 
if model_name == 'bloom':
    qllm_model, tokenizer, key = qllm_load_pretrained_from_size(model_name, model_size)
elif model_name == 'opt':
    # use converted weight
    path = os.path.join(target_storage_folder, f"{model_name}_{model_size}")
    if not os.path.exists(path):
        raise ValueError("Please run weight_convert.py first")
    qllm_model, tokenizer, key = qllm_load_pretrained_from_size(model_name, model_size, target_storage_folder=target_storage_folder)
loaded_llm_cpu = qllm_model

# sharding strategies
# load the strategy generated by the SHAQ
method = args.method
sol_file = f"{args.strat_file_name}.pkl"
root_dir = os.environ['ROOT_DIR']
strat_folder = f'{root_dir}/scripts/part_strategy'
sols_path = f'{strat_folder}/{sol_file}'
sols = pickle.load(open(sols_path, "rb"))
num_tokens_to_generate = sols['mu_n']
max_tokens_to_generate = sols['n']
bs_token = sols['gloabl_bz'] # how many sentence in a batch
assert args.method in sols, f"no {args.method} in {sols_path}"
# get sols info
sol = sols[args.method]
sharding_strategy = sol['use_plan']
print(sharding_strategy)
prefill_bs = sol['prefill_bz']
decoder_bss = sol['bz_decode_bss']
prompt_length = args.prompt_length if sols.get('prompt_length') is None else sols['prompt_length']

partition_result = sol['plan']['partition_result']
# start doing partition
state_dict = loaded_llm_cpu.state_dict()
keys = list(state_dict.keys())
# model.decoder.layers.0.  - x
shards = [{} for _ in range(len(partition_result))]
if model_name == 'opt':
    for shard_id, partition_begin_end in partition_result.items():
        begin, end = partition_begin_end
        for layer_idx in range(begin, end):
            key = f'model.decoder.layers.{layer_idx}'
            for k_candidate in keys:
                if key in k_candidate:
                    # update the shard
                    shards[shard_id][k_candidate] = state_dict.pop(k_candidate)
                    keys = list(state_dict.keys())
# save the shards and non-shards
# save the shards
for shard_id, shard in enumerate(shards):
    path = os.path.join(target_storage_folder, f"sharded_decode_weight_{shard_id}.pt")
    torch.save(shard, path) 
# save the non-shards
path = os.path.join(target_storage_folder, f"non_decoder_weight.pt")
torch.save(state_dict, path)
