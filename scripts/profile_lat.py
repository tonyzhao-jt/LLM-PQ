# profile latency
# load an OPTSharded Decoder
from qpipe.profiler import profile_lat
from qpipe.utils import get_device_name
from qllm.models.OPT.opt import model_cards
from qllm.models.OPT.seq_layers import OPTDecoderLayerSharded
from qllm import get_available_bits
import os 
import argparse
import pandas as pd 
import copy 
def parse_args():
    parser = argparse.ArgumentParser(description='Profile a transformer model')
    parser.add_argument('--model-size', type=str, default='175b', help='Size of the transformer model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size range(1,32)')
    parser.add_argument('--input-seq-length', type=int, default=1, help='Length of input sequence')
    parser.add_argument('--past-seq-length', type=int, default=2048, help='Length of past sequence')
    parser.add_argument('--generated-seq-length', type=int, default=1, help='Length of generated sequence')
    parser.add_argument('--step', type=int, default=1, help='Profiled step')
    parser.add_argument('--repeat', type=int, default=100, help='Number of iterations to profile')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--bit', type=str, default='8:tc', help='Precision bit setting')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # Access the hyperparameters
    model_size = args.model_size
    batch_size = args.batch_size
    input_seq_length = args.input_seq_length
    past_seq_length = args.past_seq_length
    repeat = args.repeat
    warmup = args.warmup
    bit = args.bit
    step = args.step 

    generated_seq_length = args.generated_seq_length
    file_path = os.path.dirname(os.path.realpath(__file__))
    device_name = get_device_name()

    file_name = device_name + "_" + str(model_size) + ".csv"
    available_bits = get_available_bits()

    config = model_cards[model_size]
    decoder_layer = OPTDecoderLayerSharded(config)
    h1 = model_cards[model_size].hidden_size
    h2 = decoder_layer.fc1.weight.shape[0]

    # print(h1, h2)

    # check whether the file exists
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame(columns=['shard', 'h1', 'h2', \
                                    'bit', 'batch_size', 'input_seq_length', 'past_seq_length', \
                                    'lat_avg', 'mem_weight', 'mem_kv', 'mem_embedding'])

    for shard in [0, 1]:
        for bz in range(1, batch_size):
            for bit in available_bits:
                for i in range(1, generated_seq_length+1, step):
                    # check if the entry has been profiled
                    if len(df[(df['shard'] == shard) & (df['h1'] == h1) & (df['h2'] == h2) & \
                                (df['bit'] == bit) & (df['batch_size'] == bz) & (df['input_seq_length'] == input_seq_length) & \
                                (df['past_seq_length'] == past_seq_length + i)]) > 0:
                        print("Entry has been profiled, skip")
                        continue
                    # profile latency
                    lat_avg, mem_weight, mem_kv, mem_embedding = profile_lat.profile_decoder_layer(config, decoder_layer, shard=shard, batch_size=bz, \
                                        input_seq_length=input_seq_length, past_seq_length=past_seq_length + i, bit=bit, \
                                            warmup=warmup, repeat=repeat, verbose=True)
                    mem_all = mem_weight + mem_kv + mem_embedding
                    df = df._append({'shard': shard, 'h1': h1, 'h2': h2, \
                                'bit': bit, 'batch_size': bz, 'input_seq_length': input_seq_length, \
                                'past_seq_length': past_seq_length + i, 'lat_avg': lat_avg, \
                                'mem_weight': mem_weight, 'mem_kv': mem_kv, 'mem_embedding': mem_embedding, \
                                'mem_all': mem_all}, ignore_index=True)
                # store the result
                df.to_csv(file_name, index=False)
    # Write the DataFrame to a CSV file
    df.to_csv(file_name, index=False)