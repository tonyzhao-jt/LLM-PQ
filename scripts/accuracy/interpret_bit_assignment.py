# finally want result like
# {0: {qkv bit, dense bit, fc1 and fc2}}
# {'self_attention.query_key_value': Linear(in_features=1024, out_features=3072, bias=True), 'self_attention.dense': Linear(in_features=1024, out_features=1024, bias=True), 'mlp.dense_h_to_4h': Linear(in_features=1024, out_features=4096, bias=True), 'mlp.dense_4h_to_h': Linear(in_features=4096, out_features=1024, bias=True)}
# load bit assignment
import pickle
strategy_result_file_path = '/workspace/qpipe/scripts/strategy/uniform_partition_bit_8_66b_Tesla_V100-SXM2-32GB_4_NVIDIA_A100-SXM4-40GB_2.pkl'
with open(strategy_result_file_path, 'rb') as f:
    strategy_result = pickle.load(f)
# bit assignment
bit_assignment = strategy_result['bit_assignment']
print(bit_assignment)