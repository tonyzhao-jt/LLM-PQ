# profile latency
# load an OPTSharded Decoder
import torch 
from qllm.utils import to_device_recursive
import lptorch
from time import perf_counter
from ..utils import get_size_cuda

# inf = float('inf')
def profile_decoder_layer(config, decoder_layer, shard=0, batch_size=1, input_seq_length=1, past_seq_length=2048, bit=8,\
                        mem_unit='MB', warmup=10, repeat=100, verbose=True):

    # construct fake input, fake KV
    h1 = config.hidden_size

    shard_strategy = {'shard': [shard], 'bits': [bit]}

    # fake input = b * 1 * hidden_size
    fake_input = torch.randn(batch_size, input_seq_length, h1)

    # fake kv size = b * (past_seq_length) * num_head, head_dims
    num_heads, head_dim = config.num_attention_heads, config.hidden_size // config.num_attention_heads

    # caliber run
    caliber = lptorch.inner_caliber
    caliber.set_model(decoder_layer)
    caliber.register_forward_hooks()
    # get calib result
    with torch.no_grad():
        decoder_layer(fake_input)
    caliber.remove_forward_hooks()

    # shard and verify the kernel
    decoder_layer = decoder_layer.to(torch.float16)  # need to first convert weight to fp16
    try:
        decoder_layer.shard(shard_strategy)
    except:
        availability_result = [False]
    availability_result = decoder_layer.verify_kernel()

    if False not in availability_result:
        if bit == '8:tc':
            input_bit = 8
        else:
            input_bit = 16
        if input_bit != 8:
            fake_k = torch.randn(batch_size, past_seq_length, num_heads, head_dim).to(torch.float16)
            fake_v = torch.randn(batch_size, past_seq_length, num_heads, head_dim).to(torch.float16)
        else:
            # int8
            fake_k = torch.randint(-128, 127, (batch_size, past_seq_length, num_heads, head_dim), dtype=torch.int8)
            fake_v = torch.randint(-128, 127, (batch_size, past_seq_length, num_heads, head_dim), dtype=torch.int8)

        fake_k.transpose_(1, 2)  # follow attn implementation
        fake_v.transpose_(1, 2)
        past_key_value = (fake_k, fake_v)

        # to fp16
        hidden_states = fake_input.to(torch.float16)
        if input_bit == 8:
            hidden_states = 127 * hidden_states
            hidden_states = hidden_states.to(torch.int8)

        # profile
        decoder_layer = decoder_layer.cuda()
        hidden_states = hidden_states.cuda()
        past_key_value = to_device_recursive(past_key_value, torch.device("cuda:0"))


        # warmup
        for i in range(warmup):
            decoder_layer(hidden_states, past_key_value=past_key_value)
            torch.cuda.synchronize()
        start = perf_counter()
        for i in range(repeat):
            decoder_layer(hidden_states, past_key_value=past_key_value)
            torch.cuda.synchronize()
        end = perf_counter()
        lat_avg = (end - start) / repeat

        # calculate weight memory, kv memory and embedding memory
        mem_weight = get_size_cuda(decoder_layer, unit=mem_unit)
        if shard == 0:
            mem_kv = get_size_cuda(past_key_value[0], unit=mem_unit) * 2
        else:
            mem_kv = 0
        mem_embedding = get_size_cuda(hidden_states, unit=mem_unit)
    else:
        inf = 99999
        lat_avg = inf # not implementable
        mem_weight, mem_kv, mem_embedding = 0, 0, 0
    caliber.clear_calib_data()
    layer_name = 'self_attn' if shard == 0 else 'ffn'
    if verbose:
        print(f"decoder_layer {layer_name} (bit={bit}): {lat_avg}")
        print(f"decoder_layer {layer_name} (bit={bit}, unit {mem_unit}): {mem_weight} {mem_kv} {mem_embedding}")
    return lat_avg, mem_weight, mem_kv, mem_embedding
