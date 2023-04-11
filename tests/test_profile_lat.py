# profile latency
# load an OPTSharded Decoder
from qpipe.profiler import profile_lat
from qllm.models.OPT.opt import model_cards
from qllm.models.OPT.seq_layers import OPTDecoderLayerSharded
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Profile a transformer model')
    parser.add_argument('--model-size', type=str, default='175b', help='Size of the transformer model')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--input-seq-length', type=int, default=1, help='Length of input sequence')
    parser.add_argument('--past-seq-length', type=int, default=2048, help='Length of past sequence')
    parser.add_argument('--repeat', type=int, default=100, help='Number of iterations to profile')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--bit', type=str, default='8:tc', help='Precision bit setting')
    parser.add_argument('--shard', type=int, default=0, choices=[0, 1], help='Sharding mode (0 for self-attention, 1 for FFN)')
    # add verbose, store_true
    parser.add_argument('--verbose', action='store_true', help='Print out the latency and memory usage')
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

    lat_avg, mem_weight, mem_kv, mem_embedding = profile_lat.profile_decoder_layer(model_size, model_cards, OPTDecoderLayerSharded, shard=shard, batch_size=batch_size, \
                        input_seq_length=input_seq_length, past_seq_length=past_seq_length, bit=bit, \
                              warmup=warmup, repeat=repeat, verbose=args.verbose)