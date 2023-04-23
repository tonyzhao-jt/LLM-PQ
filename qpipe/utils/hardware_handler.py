
import sys
import os

### Device info
DEVICE_INFO = {
        "T4": {
            "SM_count": 40,
            "max_block": 16,
            "max_reg": 64 * 1024,
            "max_shmem": 64 * 1024,
            "max_warp": 32,
            "capability": 75,
            "fp32_au_per_sm": 64,
            "tensor_core_au_per_sm": 8,
            "BW_GB_per_s": 320,
            "clock_MHz": 1590
        },
        "P4": {
            "SM_count": 20,
            "max_block": 32,
            "max_reg": 64 * 1024,
            "max_shmem": 96 * 1024,
            "max_warp": 64},
        "V100": {
            "SM_count": 80,
            "max_block": 32,
            "max_reg": 64 * 1024,
            "max_shmem": 96 * 1024,
            "max_warp": 64,
            "BW_GB_per_s": 900,
            "tensor_tflops": 125,
            "fp16_tflops": 125,
            "fp32_tflops": 15.7,
            "capability": 70,
            "fp32_au_per_sm": 64,
            "tensor_core_au_per_sm": 8,
            "clock_MHz": 1530
        },
        "A30": {
            "SM_count": 56,
            "max_block": 32,
            "max_reg": 64 * 1024,
            "max_shmem": 164 * 1024,
            "max_warp": 64,
            "BW_GB_per_s": -1
        },
        "A10": {
            "SM_count": 72,
            "max_block": 32,
            "max_reg": 64 * 1024,
            "max_shmem": 164 * 1024,
            "capability": 80, # actually 86
            "max_warp": 64},
        "A100": {
            "SM_count": 108,
            "max_block": 32,
            "max_reg": 64 * 1024,
            "max_shmem": 164 * 1024,
            "max_warp": 64,
            "BW_GB_per_s": 1600,
            "fp16_tflops": 312,
            "fp32_tflops": 19.5,
            "capability": 80
        },
        "A100-SXM4-40GB_1g_5gb":{
            "BW_GB_per_s": 3 * 1600 / 8
        }, 
        "A100-SXM4-40GB_2g_10gb": {
            "BW_GB_per_s": 2 * 1600 / 8
        }, 
        "A100-SXM4-40GB_3g_20gb": {
            "BW_GB_per_s": 4 * 1600 / 8
        },
        "A100-SXM4-40GB_4g_20gb": {
            "BW_GB_per_s": 8 * 1600 / 8
        },
        ### TODO (huhanpeng) MIG devices
    }

ALIAS_DICT = {
    "A100": ["A100-SXM4-40GB", "NVIDIA_A100-SXM4-40GB", "A100_Graphics_Device", "A100-SXM-80GB"],
    "V100": ["Tesla_V100-SXM2-32GB", "Tesla_V100-SXM2-16GB"],
    "T4": ["Tesla_T4"],
    "A100-SXM4-40GB_1g_5gb": [],
    "A100-SXM4-40GB_2g_10gb": [],
    "A100-SXM4-40GB_3g_20gb": [],
    "A100-SXM4-40GB_4g_20gb": [],
    "A30": [],
    "A10": ['NVIDIA_A10G']
}

ALIAS2SHORT = {}

for gpu_model, alias in ALIAS_DICT.items():
    ALIAS2SHORT[gpu_model] = gpu_model
    for alia in alias:
        ALIAS2SHORT[alia] = gpu_model

### **NOTE**: ALL_GPU_MODEL must be modified manually, instead of being generated from ALIAS2SHORT
# Be careful to change the order of gpu names, because changing the order of a gpu model 
# name may affect the current cached profiled data
ALL_GPU_MODEL = [
    "A100-SXM4-40GB_1g_5gb",
    "A100-SXM4-40GB_2g_10gb",
    "A100-SXM4-40GB_3g_20gb",
    "A100_Graphics_Device",
    "A100-SXM4-40GB_4g_20gb",
    "A100-SXM4-40GB",
    "Tesla_T4",
    "A30",
    "Tesla_V100-SXM2-32GB",
    "A100",
    "T4",
    "V100",
    'A10'
]

def gpu_model2int(gpu_model):
    return ALL_GPU_MODEL.index(gpu_model)

def short_device_name(gpu_model):
    return ALIAS2SHORT[gpu_model]

def query_cc(gpu_model):
    ''' Query the compute capability '''
    return DEVICE_INFO[short_device_name(gpu_model)]["capability"]

def query_bandwidth(gpu_model):
    ''' Query the bandwidth '''
    return DEVICE_INFO[short_device_name(gpu_model)]["BW_GB_per_s"]

def query_core_num(gpu_model, dtype, tensor_core_or_not):
    ''' Return the number of arithmetic units corresponding to `dtype` '''
    if dtype == "fp16":
        if tensor_core_or_not:
            return DEVICE_INFO[short_device_name(gpu_model)]["tensor_core_au_per_sm"]
        elif "fp16_au_per_sm" in DEVICE_INFO[short_device_name(gpu_model)]:
            return DEVICE_INFO[short_device_name(gpu_model)]["fp16_au_per_sm"]
        else:
            return DEVICE_INFO[short_device_name(gpu_model)]["fp32_au_per_sm"]
    elif dtype == "fp32":
        return DEVICE_INFO[short_device_name(gpu_model)]["fp32_au_per_sm"]

def get_capability():
    device = os.popen('(CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name --format=csv,noheader | tr " " _)').readlines()
    device_0 = device[0].strip()
    capability = query_cc(device_0)
    return capability

def get_device_name():
    device = os.popen('(CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name --format=csv,noheader | tr " " _)').readlines()
    device_0 = device[0].strip()
    return device_0


def get_device_mem_offline(device_name, unit='MB'):
    device_name = device_name.upper()
    mem_table = {
        "A100-SXM4-40GB": 39.44 * 1024, # not exactly = 40 * 1024
        'TESLA_T4': 14.76 * 1024,
        'TESLA_V100-SXM2-32GB': 31.74 * 1024,
        'TESLA_V100-SXM2-16GB': 14.76 * 1024,
        'NVIDIA_A100-SXM4-40GB': 39.44 * 1024,
        'A100-SXM-80GB': 79.35 * 1024,
    }
    if device_name in mem_table:
        mem = mem_table[device_name]
    else:
        raise ValueError("Unknown device name: {}".format(device_name))
    if unit == 'GB':
        mem = mem / 1024
    return mem


def get_device_name_and_mem():
    import torch
    from qllm import get_available_bits
    device_name = get_device_name()
    device_mem = torch.cuda.get_device_properties(0).total_memory
    return device_name, device_mem, get_available_bits()


def get_cuda_occupation_by_command():
    import subprocess
    gpu_index = 0 # replace with the index of the GPU you want to monitor
    output = subprocess.check_output(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits --id={gpu_index}", shell=True)
    memory_used = int(output.decode().strip().split("\n")[1])
    print(f"Memory used by GPU {gpu_index}: {memory_used} MB")