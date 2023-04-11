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
def parse_args():
    parser = argparse.ArgumentParser(description='Profile a transformer model')
    parser.add_argument('--model-size', type=str, default='175b', help='Size of the transformer model')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--input-seq-length', type=int, default=1, help='Length of input sequence')
    parser.add_argument('--past-seq-length', type=int, default=2048, help='Length of past sequence')
    parser.add_argument('--generated-seq-length', type=int, default=1, help='Length of generated sequence')
    parser.add_argument('--step', type=int, default=1, help='Profiled step')
    parser.add_argument('--repeat', type=int, default=100, help='Number of iterations to profile')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--bit', type=str, default='8:tc', help='Precision bit setting')
    parser.add_argument('--shard', type=int, default=0, choices=[0, 1], help='Sharding mode (0 for self-attention, 1 for FFN)')
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
    shard = args.shard
    step = args.step 

    generated_seq_length = args.generated_seq_length
    # store file path:
    # store the profiled result under the same folder
    file_path = os.path.dirname(os.path.realpath(__file__))
    device_name = get_device_name()
    # file name: devicename_modelsize_bs_pastseqlength_generatedtokens.csv
    # past seq length usually act as the prompts. 
    # batch_size usually fixed for a micro batch.
    shard_name = "self-attention" if shard == 0 else "FFN"
    file_name = device_name + "_" + shard_name + "_" + str(model_size) + "_" + str(batch_size) + "_" + str(past_seq_length) + "_" + str(generated_seq_length) + ".csv"
    available_bits = get_available_bits()

    # check whether the file exists
    if os.path.exists(file_name):
        print("File exists, skip profiling")
        exit(0)

    results = []
    for bit in available_bits:
        for i in range(1, generated_seq_length+1, step):
            # profile latency
            lat_avg, mem_weight, mem_kv, mem_embedding = profile_lat.profile_decoder_layer(model_size, model_cards, OPTDecoderLayerSharded, shard=shard, batch_size=batch_size, \
                                input_seq_length=input_seq_length, past_seq_length=past_seq_length + i, bit=bit, \
                                    warmup=warmup, repeat=repeat, verbose=True)
            results.append([bit, i, lat_avg, mem_weight, mem_kv, mem_embedding])
    df = pd.DataFrame(results, columns=['Bit', 'Seq Length', 'Latency (ms)', 'Memory Weight (MB)', 'Memory KV (MB)', 'Memory Embedding (MB)'])
    # Write the DataFrame to a CSV file
    df.to_csv(file_name, index=False)