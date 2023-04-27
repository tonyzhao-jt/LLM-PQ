# finally want result like
# {0: {qkv bit, dense bit, fc1 and fc2}}
# {'self_attention.query_key_value': Linear(in_features=1024, out_features=3072, bias=True), 'self_attention.dense': Linear(in_features=1024, out_features=1024, bias=True), 'mlp.dense_h_to_4h': Linear(in_features=1024, out_features=4096, bias=True), 'mlp.dense_4h_to_h': Linear(in_features=4096, out_features=1024, bias=True)}
# load bit assignment
import pickle

keys = ['self_attention.query_key_value', 
        'self_attention.dense', 
         'mlp.dense_h_to_4h',
         'mlp.dense_4h_to_h']

available_methods = [
    'pipeedge', 'qpipe', 'uniform_2', 'uniform_4', 'adabit'
]
model_size = '30b'
device_info = 'Tesla_T4_2_Tesla_V100-SXM2-32GB_1'

for method in available_methods:
    strategy_result_file_path = f'/workspace/qpipe/scripts/part_strategy/{method}_{model_size}_{device_info}_final_strategy.pkl'
    with open(strategy_result_file_path, 'rb') as f:
        strategy_result = pickle.load(f)
    # bit assignment
    # former two are for qkv and dense, latter two are for fc1 and fc2
    # first unpack the shard strategy from ranks
    pure_shard_strategy = {}
    for rank, val in strategy_result.items():
        for layer, layer_shard in val.items():
            if layer not in pure_shard_strategy:
                pure_shard_strategy.update({layer: layer_shard})
            else: 
                pure_shard_strategy[layer]['bits'].append(layer_shard['bits'][0])
                # print(pure_shard_strategy[layer]['bits'])

    # then unpack the bit assignment from shard strategy
    bit_assignment = {}
    for layer_idx, val in pure_shard_strategy.items():
        bit_assignment[layer_idx] = []
        bits = val['bits']
        int_bits = []
        for bit in bits:
            if bit == '8:tc' or bit == '8:tc-li':
                int_bits.append(8)
            else:
                int_bits.append(int(bit))
        bits = int_bits
        bit_assignment[layer_idx].append(bits[0])
        bit_assignment[layer_idx].append(bits[1])

    from qpipe.utils import (
        save_with_pickle,
    )

    folder_name = 'bit_for_gptq_test'
    file_name = f'{method}_{model_size}_{device_info}_acc_test.pkl'
    save_with_pickle(bit_assignment, file_name, folder_name)

# # load 
# import pickle
# folder_name = 'bit_for_gptq_test'
# file_name = f'{method}_{model_size}_Tesla_T4_2_Tesla_V100-SXM2-32GB_1_acc_test.pkl'
# with open(f'{folder_name}/{file_name}', 'rb') as f:
#     bit_assignment = pickle.load(f)