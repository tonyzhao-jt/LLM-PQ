import argparse
import qpipe
def simple_model_info_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='125m')
    parser.add_argument('--model-name', type=str, default='opt')
    parser.add_argument('--device-info', type=str, default=None)
    parser.add_argument('--sol-folder', type=str, default='/workspace/qpipe/scripts/part_strategy')
    args = parser.parse_args()
    # available method
    args.available_methods = ['adabits', 'adaqpipe', 'pipeedge', 'uniform']
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

def get_available_candidate_bits():
    return qpipe._globals.AVAILABLE_BITS