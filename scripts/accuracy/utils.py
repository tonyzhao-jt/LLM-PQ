import argparse
def simple_model_info_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='125m')
    parser.add_argument('--model-name', type=str, default='opt')
    args = parser.parse_args()
    return args

def model_config_and_decoder_layers(model_name, model_size):
    # load the data
    if model_name == 'opt':
        from qllm.models.OPT.opt import model_cards, get_available_models
        config = model_cards[model_size]
        num_layers = config.num_hidden_layers
    if model_name == 'bloom':
        from qllm.models.BLOOM.bloom import model_cards, get_available_models
        config = model_cards[model_size]
        num_layers = config.n_layer
    return config, num_layers