import argparse
import llm_pq
import os 
def simple_model_info_parser():
    root_folder = os.environ.get('ROOT_DIR') if os.environ.get('ROOT_DIR') else '/workspace/qpipe'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='125m')
    parser.add_argument('--model-name', type=str, default='opt')
    parser.add_argument('--device-info', type=str, default=None)
    parser.add_argument('--sol-folder', type=str, default=f'{root_folder}/scripts/part_strategy')
    parser.add_argument('--user_abs_file_path', type=str, default='')
    args = parser.parse_args()
    # available method
    args.available_methods = ['adabits', 'llm_pq', 'pipeedge', 'uniform']
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
    return llm_pq._globals.AVAILABLE_BITS