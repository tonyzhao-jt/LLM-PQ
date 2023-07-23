# profile latency
# load an OPTSharded Decoder
import torch 
from qllm.utils import to_device_recursive
import lptorch
from time import perf_counter
import time 
from ..utils import get_size_cuda, get_iter_variable_size
import copy
import numpy as np

from qllm.models import return_config_name
from qllm.models.BLOOM.utils import build_alibi_tensor, _prepare_attn_mask
import torch.nn as nn

class DecoderStacked(nn.Module):
    def __init__(self, decoder_layer, num_layers, model_type='opt'):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.model_type = model_type
    
    @torch.inference_mode()
    def forward(self, hidden_states, attention_mask=None, alibi=None):
        if self.model_type == 'opt':
            for layer in self.layers:
                hidden_states = layer(hidden_states)[0]
        elif self.model_type == 'bloom':
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask, alibi=alibi)[0]
        return hidden_states

def remove_outliers(latencies, threshold=3):
    """Remove outlier latencies using the Z-score method."""
    latencies = np.array(latencies)
    z_scores = np.abs((latencies - np.mean(latencies)) / np.std(latencies))
    filtered_latencies = latencies[z_scores < threshold]
    return filtered_latencies

# inf = float('inf')
def profile_decoder_layer(config, decoder_layer, shard=0, batch_size=1, input_seq_length=1, past_seq_length=2048, bit=8,\
                        mem_unit='MB', warmup=10, repeat=100, verbose=True):

    decoder_layer = copy.deepcopy(decoder_layer)
    # construct fake input, fake KV
    h1 = config.hidden_size

    shard_strategy = {'shard': [shard], 'bits': [bit]}
    if shard == 2:
        shard_strategy = {'shard': [0, 1], 'bits': [bit, bit]}

    # fake input = b * 1 * hidden_size
    fake_input = torch.randn(batch_size, input_seq_length, h1)

    model_name = return_config_name(config)

    # # caliber run
    caliber = lptorch.inner_caliber
    caliber.set_model(decoder_layer)
    caliber.register_forward_hooks()

    

    # print(config_name, config)

    if model_name == 'bloom':
        #atten_mask: (batch_size, max_seq_len) 
        num_heads = config.num_attention_heads
        attention_mask = torch.ones((batch_size, input_seq_length))
        alibi = build_alibi_tensor(attention_mask, num_heads, dtype=torch.float32)
        causal_mask = _prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, input_seq_length),
            past_key_values_length=0,
        )
    
    # get calib result
    with torch.no_grad():
        if model_name == 'bloom':
            decoder_layer(fake_input, attention_mask=causal_mask, alibi=alibi)
        else:
            decoder_layer(fake_input)
    caliber.remove_forward_hooks()
 
    # shard and verify the kernel
    decoder_layer = decoder_layer.to(torch.float16)  # need to first convert weight to fp16
    try:
        decoder_layer.shard(shard_strategy)
        # print(decoder_layer)
    except:
        availability_result = [False]
    
    if model_name.lower() != 'bloom':
        availability_result = decoder_layer.verify_kernel()
    else:
        availability_result = [True]

    if False not in availability_result:
        torch_dtype = torch.float16
        input_bit = 16
        # if bit == '8:tc':
        #     input_bit = 8
        #     torch_dtype = torch.int8

        hidden_states = fake_input.to(torch.float16)
        # if input_bit == 8:
        #     hidden_states = 127 * hidden_states
        #     hidden_states = hidden_states.to(torch.int8)
        
        hidden_states = hidden_states.cuda()
        decoder_layer = decoder_layer.cuda()
        if 0 in shard_strategy['shard']:
            # init kv cache 
            request_id = 1
            if hasattr(decoder_layer, 'self_attention'):
                attention_mod = decoder_layer.self_attention
            else:
                attention_mod = decoder_layer.self_attn
            
            attention_mod.init_kv_cache(batch_size, past_seq_length, input_seq_length, request_id, \
                                        torch_dtype=torch_dtype, init_with_xavier=True)
            attention_mod.profile = True # set profile to make the kv didn't increase
            attention_mod.kv_status[request_id][0] = past_seq_length
            attention_mod.kv_status[request_id][1] = input_seq_length

        
        num_stacks = 4
        decoder_stacked = DecoderStacked(decoder_layer, num_stacks, model_type=model_name)
        with torch.no_grad():
            if model_name.lower() == 'opt':
                # Warmup
                for i in range(warmup):
                    decoder_stacked(hidden_states)

                torch.cuda.synchronize()
                # start = perf_counter()
                start = time.time()
                for i in range(repeat):
                    decoder_stacked(hidden_states)
                torch.cuda.synchronize()
                # end = perf_counter()
                end = time.time()
                lat_avg = (end - start) / num_stacks / repeat * 1000 # in ms
                # # Measure latency
                # latencies = []
                # for i in range(repeat):
                #     torch.cuda.synchronize()
                #     start = perf_counter()
                #     decoder_layer(hidden_states)
                #     torch.cuda.synchronize()
                #     end = perf_counter()
                #     latencies.append(end - start)
            else:
                causal_mask = causal_mask.cuda().bool()
                alibi = alibi.cuda().to(torch_dtype)
                # Warmup
                for i in range(warmup):
                    decoder_layer(hidden_states, attention_mask=causal_mask, alibi=alibi)
                torch.cuda.synchronize()
                # Measure latency
                start = time.time()
                for i in range(repeat):
                    decoder_layer(hidden_states, attention_mask=causal_mask, alibi=alibi)
                torch.cuda.synchronize()
                # end = perf_counter()
                end = time.time()
                lat_avg = (end - start) / repeat * 1000
                # latencies = []
                # for i in range(repeat):
                #     torch.cuda.synchronize()
                #     start = perf_counter()
                #     decoder_layer(hidden_states, attention_mask=causal_mask, alibi=alibi)
                #     torch.cuda.synchronize()
                #     end = perf_counter()
                #     latencies.append(end - start)
            

        # Remove outliers and calculate average latency
        # latencies_without_outliers = remove_outliers(latencies)
        # lat_avg = sum(latencies_without_outliers) / len(latencies_without_outliers)

        # calculate weight memory, kv memory and embedding memory
        mem_weight = get_size_cuda(decoder_layer, unit=mem_unit)
        if 0 in shard_strategy['shard']:
            mem_kv = get_iter_variable_size(attention_mod.kv_cache, unit=mem_unit) * 2
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
    del decoder_stacked
    if 0 in shard_strategy['shard']:
        del attention_mod
    del fake_input
    if False not in availability_result:
        del hidden_states
    torch.cuda.empty_cache()

    return lat_avg, mem_weight, mem_kv, mem_embedding
