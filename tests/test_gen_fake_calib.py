
from qllm.models.OPT import OPTForCausalLMSeq, OPTDecoderSeq, OPTDecoderLayerSharded
from qllm.models.OPT.opt import model_cards

import lptorch
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add model size
    parser.add_argument('--model_size', type=str, default='125m', help='model size')
    args = parser.parse_args()
    model_size = args.model_size.lower()
    # fake input = b * 1 * hidden_size
    config = model_cards[model_size]
    batch_size = 10
    input_seq_length = 1
    h1 = config.hidden_size
    fake_input = torch.randn(batch_size, input_seq_length, h1)
    decoder_layer = OPTDecoderLayerSharded(config)

    caliber = lptorch.inner_caliber
    caliber.set_fake()  

    caliber.set_model(decoder_layer)
    # caliber.default_hook = caliber.torch_int_forward_hook
    caliber.register_forward_hooks()
    # get calib result
    with torch.no_grad():
        decoder_layer(fake_input)

    caliber.remove_forward_hooks()
    for layer_name, calib_res in caliber.collected_calib_data.items():
        calib_shape = caliber.collected_input_shape[layer_name]
        caliber.set_fake_module_calib_data(layer_name, calib_shape, calib_res)
    caliber.save_fake_calib_data(f'fake_calib_{model_size}.pkl')
    # caliber.load_fake_calib_data(f'fake_calib_{model_size}.pkl')

    

    


