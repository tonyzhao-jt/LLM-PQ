MEM_UNIT='MB'
CUDA_CONTEXT_MEM = 430 + 1500 # 430MB cuda context allocation + 1.5 G Torch Temp Allocation
RATIO_AVOID_OOM = 0.95 # 95% of the memory is used to avoid OOM
TIME_MULT_TIMES = 1
SLO_RATE = 1.5 # times of fp16 inference time to be SLO.
# global setup for available bits
AVAILBLE_BITS=[2, 3, 4, 8, '8:tc-li', 16]