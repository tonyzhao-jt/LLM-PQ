
from qllm.models.OPT import OPTDecoderLayerSharded
from qllm.models import opt

from qllm.models.BLOOM import BloomBlockSharded, build_alibi_tensor, _prepare_attn_mask
from qllm.models import bloom

import argparse
import lptorch
import torch
if __name__ == '__main__':
    # allow user to set model type, argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='125m', help='model size')
    parser.add_argument('--model-name', type=str, default='opt', help='model name')
    args = parser.parse_args()
    model_name = args.model_name.lower()
    if model_name == 'opt':
        model_cards = opt.model_cards
        decoder_constructor = OPTDecoderLayerSharded
    elif model_name == 'bloom':
        model_cards = bloom.model_cards
        decoder_constructor = BloomBlockSharded
    model_size = args.model_size.lower()
    # fake input = b * 1 * hidden_size
    assert model_size in model_cards, f"model size {model_size} not available, available models: {model_cards.keys()}"
    config = model_cards[model_size]
  
    batch_size = 10
    input_seq_length = 1

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

    h1 = config.hidden_size
    fake_input = torch.randn(batch_size, input_seq_length, h1)
    decoder_layer = decoder_constructor(config)

    caliber = lptorch.inner_caliber
    caliber.set_fake()  

    caliber.set_model(decoder_layer)
    # caliber.default_hook = caliber.torch_int_forward_hook
    caliber.register_forward_hooks()
    # get calib result
    with torch.no_grad():
        if model_name == 'bloom':
            decoder_layer(fake_input, attention_mask=causal_mask, alibi=alibi)
        else:
            decoder_layer(fake_input)
    caliber.remove_forward_hooks()
    for layer_name, calib_res in caliber.collected_calib_data.items():
        calib_shape = caliber.collected_input_shape[layer_name]
        caliber.set_fake_module_calib_data(layer_name, calib_shape, calib_res)

    caliber.save_fake_calib_data(f'fake_calib_{model_name}_{model_size}.pkl')
    caliber.load_fake_calib_data(f'fake_calib_{model_name}_{model_size}.pkl')
    print("calib generated for model", model_name, model_size)