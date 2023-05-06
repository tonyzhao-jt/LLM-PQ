# profile latency
# load an OPTSharded Decoder
import torch 
from qllm.utils import to_device_recursive
import lptorch
from time import perf_counter
from ..utils import get_size_cuda, get_iter_variable_size
import copy

# inf = float('inf')
def profile_decoder_layer(config, decoder_layer, shard=0, batch_size=1, input_seq_length=1, past_seq_length=2048, bit=8,\
                        mem_unit='MB', warmup=10, repeat=100, verbose=True):

    decoder_layer = copy.deepcopy(decoder_layer)
    # construct fake input, fake KV
    h1 = config.hidden_size

    shard_strategy = {'shard': [shard], 'bits': [bit]}

    # fake input = b * 1 * hidden_size
    fake_input = torch.randn(batch_size, input_seq_length, h1)

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
        torch_dtype = torch.float16
        input_bit = 16
        # if bit == '8:tc':
        #     input_bit = 8
        #     torch_dtype = torch.int8

        hidden_states = fake_input.to(torch.float16)
        if input_bit == 8:
            hidden_states = 127 * hidden_states
            hidden_states = hidden_states.to(torch.int8)
        
        hidden_states = hidden_states.cuda()
        decoder_layer = decoder_layer.cuda()
        if 0 in shard_strategy['shard']:
            # init kv cache 
            decoder_layer.self_attention.init_kv_cache(batch_size, past_seq_length, input_seq_length, 1, torch_dtype=torch_dtype)
            decoder_layer.self_attention.profile = True # set profile to make the kv didn't increase 

        # warmup
        for i in range(warmup):
            decoder_layer(hidden_states)
            torch.cuda.synchronize()
        start = perf_counter()
        for i in range(repeat):
            decoder_layer(hidden_states)
            torch.cuda.synchronize()
        end = perf_counter()
        lat_avg = (end - start) / repeat

        # calculate weight memory, kv memory and embedding memory
        mem_weight = get_size_cuda(decoder_layer, unit=mem_unit)
        if 0 in shard_strategy['shard']:
            mem_kv = get_iter_variable_size(decoder_layer.self_attention.kv_cache, unit=mem_unit) * 2
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
    
    # del all the instances
    del decoder_layer
    del fake_input
    if False not in availability_result:
        del hidden_states
    torch.cuda.empty_cache()

    return lat_avg, mem_weight, mem_kv, mem_embedding
