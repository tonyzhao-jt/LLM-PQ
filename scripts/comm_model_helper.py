import argparse
from shaq.partitioner.helper import get_device_info
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, required=True)
parser.add_argument('--device_names',  nargs='+', type=str, required=True)
parser.add_argument('--device_numbers',  nargs='+', type=int, required=True)
args = parser.parse_args()

# model size
model_size = args.model_size # '66b'
device_names = args.device_names # ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
device_numbers = args.device_numbers # [2, 3]
assert len(device_names) == len(device_numbers), "device_names and device_numbers should have the same length"
device_info = get_device_info(device_names, device_numbers)

import os
comm_cost_model_dir = f'/workspace/qpipe/scripts/comm_cost_model/{device_info}'
# if folder not exists, make one
if not os.path.exists(comm_cost_model_dir):
    os.mkdir(comm_cost_model_dir)