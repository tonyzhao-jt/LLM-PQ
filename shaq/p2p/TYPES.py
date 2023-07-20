import torch 
import collections
# Base tag values
TAG_BASE_DATA = 0
TAG_BASE_CMD = 10

# Offsets which are added to base values above
TAG_TENSOR_COUNT = 0
TAG_TENSOR_DTYPE_SHAPELEN = 1
TAG_TENSOR_SHAPE = 2
TAG_TENSOR = 3
TAG_TENSOR_PICKLED_SIZE = 4

TORCH_TYPES = [ torch.float32,
                torch.float64,
                torch.complex64,
                torch.complex128,
                torch.float16,
                torch.bfloat16,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.bool ]

TORCH_TYPES_ENUM = collections.OrderedDict()
for i, t in enumerate(TORCH_TYPES):
    TORCH_TYPES_ENUM[t] = i