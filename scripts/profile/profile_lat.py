import os 
import argparse
import pandas as pd 

import qllm.utils as qllm_utils
from qllm.models import create_empty_decoder

import shaq
from shaq.logger import logger
from shaq.profiler import profile_lat
from shaq.utils import get_device_name_and_mem


def parse_args():
    parser = argparse.ArgumentParser(description='Profile a transformer model')
    parser.add_argument('--model-name', type=str, default='opt', help='model name')
    parser.add_argument('--model-size', type=str, default='175b', help='Size of the transformer model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size range(1,32)')
    parser.add_argument('--input-seq-length', type=int, default=1, help='Length of input sequence')
    parser.add_argument('--past-seq-length', type=int, default=2048, help='Length of past sequence')
    parser.add_argument('--generated-seq-length', type=int, default=1, help='Length of generated sequence')
    parser.add_argument('--step', type=int, default=1, help='Profiled step')
    parser.add_argument('--repeat', type=int, default=100, help='Number of iterations to profile')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--bit', type=str, default='8:tc', help='Precision bit setting')
    parser.add_argument('--num-stacks', type=int, default=4, help='Number of repeat stacks')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # Access the hyperparameters
    model_name = args.model_name
    model_size = args.model_size
    batch_size = args.batch_size
    input_seq_length = args.input_seq_length
    past_seq_length = args.past_seq_length
    repeat = args.repeat
    warmup = args.warmup
    bit = args.bit
    step = args.step 
    num_stacks = args.num_stacks

    generated_seq_length = args.generated_seq_length
    folder_name = 'lat_profiled_result' 
    file_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_path, folder_name)

    device_name, device_mem, _ = get_device_name_and_mem()
    available_bits = shaq._globals.AVAILABLE_BITS

    file_name = device_name + "_" + str(model_size) + ".csv"

    decoder_layer, (h1, h2), config = create_empty_decoder(model_name, model_size)
    # must set, else bitsandbyyes profiled wrongly
    # np_weight_folder_path = os.environ.get('NP_WEIGHT_FOLDER', False)
    # assert np_weight_folder_path, "Please set NP_WEIGHT_FOLDER"
    # weight_path = os.path.join(np_weight_folder_path,f"{model_name}_{model_size}") 
    # assert os.path.exists(weight_path), f"Please convert the weight file to {weight_path}"
    # layer_idx = 6 # randomly set
    # if model_name == 'opt':
    #     decoder_layer = qllm_utils.load_np_weight_opt_layer(weight_path, layer_idx, decoder_layer)
    # elif model_name == 'bloom':
    #     decoder_layer, = qllm_utils.load_np_weight_bloom_layer(weight_path, layer_idx, decoder_layer)
    if model_name == 'opt':
        np_weight_folder_path = os.environ.get('NP_WEIGHT_FOLDER', False)
        assert np_weight_folder_path, "Please set NP_WEIGHT_FOLDER"
        weight_path = os.path.join(np_weight_folder_path,f"{model_name}_{model_size}") 
        assert os.path.exists(weight_path), f"Please convert the weight file to {weight_path}"
        layer_idx = 6 # randomly set
        decoder_layer = qllm_utils.load_np_weight_opt_layer(weight_path, layer_idx, decoder_layer)
    elif model_name == 'bloom':
        # decoder_layer, = qllm_utils.load_np_weight_bloom_layer(weight_path, layer_idx, decoder_layer)
        # decoder_layer, = qllm_utils.load_np_weight_bloom_layer(weight_path, layer_idx, decoder_layer)
        logger.error("bloom temporarily not able to load ")
    decoder_layer = decoder_layer.float()
    # print(h1, h2)
    def convert_to_int(x):
        if type(x) is float:
            return int(x)
        elif type(x) is str and x.isnumeric():
            return int(float(x))
        else:
            return x

    # check whether the file exists
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame(columns=['shard', 'h1', 'h2', \
                                    'bit', 'batch_size', 'input_seq_length', 'past_seq_length', \
                                    'lat_avg', 'mem_weight', 'mem_kv', 'mem_embedding'])

    for shard in [2]:
        # for bz in range(1, batch_size + 1):
        for bit in available_bits:
            bz = batch_size
            for i in range(0, generated_seq_length + 1, step):
                # i = generated_seq_length
                # check if the entry has been profiled
                if len(df[(df['shard'] == shard) & (df['h1'] == h1) & (df['h2'] == h2) & \
                            (df['bit'].astype(str) == str(bit)) & (df['batch_size'] == bz) & (df['input_seq_length'] == input_seq_length) & \
                            (df['past_seq_length'] == past_seq_length + i)]) > 0:
                    logger.info("Entry has been profiled, skip")
                    continue
                # profile latency
                lat_avg, mem_weight, mem_kv, mem_embedding = profile_lat.profile_decoder_layer(config, decoder_layer, shard=shard, batch_size=bz, \
                                    input_seq_length=input_seq_length, past_seq_length=past_seq_length + i, bit=bit, \
                                        warmup=warmup, repeat=repeat, verbose=True, num_stacks=num_stacks)
                mem_all = mem_weight + mem_kv + mem_embedding
                # not available for pandas < 1.3.5
                # check pandas version to select which code piece
                if pd.__version__ > '1.3.5':
                    df = df._append({'shard': shard, 'h1': h1, 'h2': h2, \
                                'bit': str(bit), 'batch_size': bz, 'input_seq_length': input_seq_length, \
                                'past_seq_length': past_seq_length + i, 'lat_avg': lat_avg, \
                                'mem_weight': mem_weight, 'mem_kv': mem_kv, 'mem_embedding': mem_embedding, \
                                'mem_all': mem_all}, ignore_index=True)
                else:
                    new_row = {'shard': shard, 'h1': h1, 'h2': h2, 'bit': str(bit), 'batch_size': bz, 'input_seq_length': input_seq_length, 'past_seq_length': past_seq_length + i, 'lat_avg': lat_avg, 'mem_weight': mem_weight, 'mem_kv': mem_kv, 'mem_embedding': mem_embedding, 'mem_all': mem_all}
                    df = df.append(new_row, ignore_index=True)
            # store the result
            df.to_csv(file_name, index=False)
    # Write the DataFrame to a CSV file
    df.to_csv(file_name, index=False)