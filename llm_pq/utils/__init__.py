from .device_handler import to_device
from .hardware_handler import (
    get_device_name, get_device_name_and_mem, get_device_mem_offline, get_cuda_occupation_by_command,
    query_cc, query_bandwidth, to_weight_int8_if_tc_not_available, has_tc
)
from .mem_handler import get_size_cuda, get_size_cpu, get_iter_variable_size
from .save import save_with_pickle
from .simple_partition import partition_a_into_b_bins, get_default_decode_bz
from .tensor_handler import object_to_tensor
from .bits_pair import get_available_bits_pair, get_available_bits_pair_decoupled
from .misc import get_factors, roundup_power2_divisions
from .D_handler import convert_D_to_ranked_device

# api server
from .random import random_uuid