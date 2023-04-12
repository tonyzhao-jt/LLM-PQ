# partitioner here
# input: cost_model.pkl, model# partitioner here
# input: cost_model.pkl, model_cfg
# output: partitioned model strategy

# read configuration from the model_cfg and one decoder layer
from qllm.models.OPT.opt import model_cards
from qllm.models.OPT.seq_layers import OPTDecoderLayerSharded
from qllm.utils import ModelMemEstimator
import numpy as np 

# read basic configuration from the model_cfg
model_size = '175b'
config = model_cards[model_size]
decoder_layer = OPTDecoderLayerSharded(config)
h1 = model_cards[model_size].hidden_size
h2 = decoder_layer.fc1.weight.shape[0]
num_hidden_layers = config.num_hidden_layers # decoder layer numbers

# set the chunk size
b = 32
# set the prompt sequence length
s = 2048
# set the number of generated tokens
n = 50

model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n)
hidden_space_numels = model_mem_estimator.estimate_hidden_space()

# Cost Model
def calculate_cost(bit, shard, batch_size, input_seq_length, past_seq_length, generated_seq_length):
    # calculate the comm
    comm = 1
    # calculate the latency
    lat = 1
    return comm, lat

def calculate_mem_requirement(bit, shard):
    partition = {0: {"shard": [shard], "bits": [bit]}}
    mem_require = model_mem_estimator.calculate_maximum_mem_occupation_of_partition(partition, unit='MB')
    return mem_require

# T equals to num_hidden_layers, 0,1
T = [0,1] * num_hidden_layers
# device is a list of device names {rank: device_name}
D = {0: 'NVIDIA_A100-SXM4-40GB', 1: 'NVIDIA_A100-SXM4-40GB', 2: 'NVIDIA_A100-SXM4-40GB', 3:'NVIDIA_A100-SXM4-40GB', \
     4: 'Tesla_V100-SXM2-32GB', 5: 'Tesla_V100-SXM2-32GB', 6: 'Tesla_V100-SXM2-32GB', 7: 'Tesla_V100-SXM2-32GB'}
# bandwidth between devices, we use cost model to calculate instead
B = {(0,1): [1, 1], (1,0): [1, 1], (0,2): [1, 1], (2,0): [1, 1], (0,3): [1, 1], (3,0): [1, 1], \
     (1,2): [1, 1], (2,1): [1, 1], (1,3): [1, 1], (3,1): [1, 1], (2,3): [1, 1], (3,2): [1, 1], \
    (4,5): [1, 1], (5,4): [1, 1], (4,6): [1, 1], (6,4): [1, 1], (4,7): [1, 1], (7,4): [1, 1], \
    (5,6): [1, 1], (6,5): [1, 1], (5,7): [1, 1], (7,5): [1, 1], (6,7): [1, 1], (7,6): [1, 1], \
    (0,4): [1, 1], (4,0): [1, 1], (1,5): [1, 1], (5,1): [1, 1], (2,6): [1, 1], (6,2): [1, 1], (3,7): [1, 1], (7,3): [1, 1]
}
# Here we first implement the PipeEdge Algorithm regarding the model partition among different devices
def partition(T, D, B, bit=16):
    L = len(T)  # Number of layers in the model
    M = [calculate_memory_requirement(layer) for layer in T]  # Memory requirements of each layer
    h = {}  # Memoization table for storing optimal time
    answer = float('inf')  # Initial value for the answer
    index = None  # Index of the optimal solution
    # Initialize memoization table
    for i in range(L):
        for S in get_subsets(D):
            for u in D - S:
                h[(i, S, u)] = float('inf')
    h[(0, frozenset(), None)] = 0
    # Compute optimal time for each layer
    for i in range(L - 1):
        for S in get_subsets(D):
            for u in D - S:
                for j in range(i + 1, L):
                    if sum(M[i:j]) > D[u]:
                        break
                    C = calculate_time(T[i:j], D - {u}, B)
                    if j == L - 1:
                        if C < answer:
                            answer = C
                            index = (L, frozenset(S), u)
                    else:
                        for v in D - S - {u}:
                            if C < h[(j, S | {u}, v)]:
                                h[(j, S | {u}, v)] = C
                                p[(j, S | {u}, v)] = (i, u)
    # Find optimal time
    Topt = float('inf')
    for S in get_subsets(D):
        Topt = min(h[(L-1, S, None)], Topt)
    # Find optimal strategy
    R = []
    (i, S, u) = index
    R.append((i+1, L, u))
    while i > 0:
        (i, u) = p[(i, S | {u}, u)]
        R.append((i+1, index[0], u))
        S -= {u}
        index = (i, S, u)
    R.reverse()
    return Topt, R

def calculate_memory_requirement(layer):
    # TODO: Calculate memory requirement for a layer
    pass

def get_subsets(s):
    # TODO: Get all subsets of a set
    pass

def calculate_time(layers, D, B):
    # TODO: Calculate time for executing a sequence of layers on a set of devices
    pass


partition(T, D, B)