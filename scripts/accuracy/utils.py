import argparse
def simple_model_info_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='125m')
    parser.add_argument('--model-name', type=str, default='opt')
    args = parser.parse_args()
    return args

    