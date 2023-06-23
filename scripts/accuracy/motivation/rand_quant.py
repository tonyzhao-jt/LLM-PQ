# take a model, given the compression rate (reduce memory required)
# given different types of inidcator, test the accuracy of them in the gptq

import qpipe
from qpipe.partitioner.helper import (
    create_mem_estimator
)
from qpipe.partitioner.utils import (
    assign_uniform_bit, 
    estimate_min_max_mem,
)

from qpipe.cost_model import (
    get_mem_with_layer_bit_pair
)

from qpipe.partitioner.indicator import (
    assign_omega_uniform,
    assign_omega_constant
)
from qpipe.utils import (
    save_with_pickle, get_available_bits_pair
)

from qpipe.logger import logger
import pickle
from qllm.models import opt
from qllm.models import bloom

import numpy as np 
# setup ilp configs
import pulp
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam('WLSACCESSID',"1b28dca7-337e-4811-b346-01087e09cd64")
env.setParam('WLSSECRET', "629520bd-a114-45d7-b828-bfc5235c198d")
env.setParam('LICENSEID', 965996)
env.start()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default='125m')
parser.add_argument('--model_name', type=str, default='opt')
parser.add_argument('--ratio', type=float, default=0.5)
# add the indicator typ
parser.add_argument('-it', '--indicator_type', type=str, default='uniform')
parser.add_argument('--file_name', type=str, default=None)
args = parser.parse_args()

unit = qpipe._globals.MEM_UNIT
time_mult_times = qpipe._globals.TIME_MULT_TIMES
# model size
model_name = args.model_name # 'opt'
model_size = args.model_size # '66b'

from qpipe.partitioner import gen_config
# generation configs
global_bz = gen_config.global_bz
micro_bz = gen_config.micro_bz
chunk_size = global_bz // micro_bz
s = gen_config.s
n = gen_config.n

if model_name == 'opt':
    assert model_size in opt.model_cards.keys(), f"model_size {model_size} not in opt.model_cards.keys()"
    config = opt.model_cards[model_size]
    num_layers = config.num_hidden_layers
elif model_name == 'bloom':
    assert model_size in bloom.model_cards.keys(), f"model_size {model_size} not in bloom.model_cards.keys()"
    config = bloom.model_cards[model_size]
    num_layers = config.n_layer

# fake T
T = [0, 1] * num_layers
model_mem_estimator = create_mem_estimator(global_bz, s, n, config)
# estimate the overall memory requried for fp16
# assign bits
bit_map = {}
indicator_type = args.indicator_type
assign_uniform_bit(T, 16, bit_map)
# initial_mem = estimate_all_layer_mem(model_mem_estimator, T, bit_map)
max_model_mem, min_model_mem = estimate_min_max_mem(model_mem_estimator, T)
# prepare the ILP problem
logger.info(f"max model mem{max_model_mem}")
logger.info(f"min model mem{min_model_mem}")

# calculate the memory required for compression, lie in [0,1]
ratio = args.ratio
M_c = memory_contraints = min_model_mem + (max_model_mem - min_model_mem) * ratio
logger.info(f"memory_contraints {memory_contraints}")

# prepare the ilp problem in bit allocation.
# problem is easy, first fetch the indicator for each layer (self attn, ff)
# then based on the setting, we calculte the optimal bit allocation that minimize the indicator result.
L = num_layers
available_bits = [3, 4, 8, 16] # regard 8-bit as same
available_bits = list(set(available_bits))
BITs = get_available_bits_pair(available_bits)
if indicator_type == 'uniform':
    omega = assign_omega_uniform(L, BITs)
elif indicator_type == 'constant':
    omega = assign_omega_constant(L, BITs)
elif indicator_type == 'mag':
    # use magnitude as the indicator
    file_name = args.file_name
    assert file_name is not None, "file_name is None"
    omega = pickle.load(open(file_name, 'rb'))

# memory constaints
mem_bits_vector = get_mem_with_layer_bit_pair(BITs, model_mem_estimator)
M = np.tile(mem_bits_vector, (L, 1))

# ilp problem
def solve_ilp_pulp(L, BITs, M, M_c, omega):
    prob = pulp.LpProblem("inidcator minimize", pulp.LpMinimize)
    # Create a new PuLP model
    B = len(BITs)
    x = pulp.LpVariable.dicts("x", [(i, b) for i in range(L) for b in range(B)], cat=pulp.LpBinary)

    # Define the objective function
    prob += pulp.lpSum([omega[(i, b)] * x[(i, b)] for i in range(L) for b in range(B)])

    for i in range(L):
        prob += pulp.lpSum([x[(i, b)] for b in range(B)]) == 1
    # simple mem constraints
    prob += pulp.lpSum([x[(i, b)] * M[(i, b)] for i in range(L) for b in range(B)]) <= M_c

    
    # Solve the problem
    # solver = pulp.GUROBI(msg=True, threads=0, timeLimit=100, MIPGap=0.003)
    solver = pulp.GUROBI()
    # solver = pulp.GUROBI(msg=True)
    prob.solve(solver)

    # Print the solution status
    print("Status:", pulp.LpStatus[prob.status])
    # Print the optimal objective value
    print("Optimal value of the objective function:", pulp.value(prob.objective))
    # store the optimal solution
    result = {}
    # print z variable result
    for i in range(L):
        for b in range(B):
            if x[(i, b)].varValue > 0:
                result[i] = b
    return result, pulp.value(prob.objective)

result, opt_value = solve_ilp_pulp(L, BITs, M, M_c, omega)
# interpret result into bit assignment
bit_assignment = {}
# block_idx = 0
# for layer, bits in result.items():
#     attn_bit, ff_bit = BITs[bits]
#     bit_assignment[block_idx] = attn_bit
#     bit_assignment[block_idx + 1] = ff_bit
#     block_idx += 2
# gptq read layer: [bit1, bit2]
for layer, bits in result.items():
    bit_assignment[layer] = BITs[bits]

print(bit_assignment)

folder_name = 'bit_result'
file_name = f'{indicator_type}_{model_name}_{model_size}_{ratio}_bit_ass.pkl'
save_with_pickle(bit_assignment, file_name, folder_name)
