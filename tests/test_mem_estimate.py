from llm_pq.partitioner.helper import create_mem_estimator
from qllm.models.OPT.opt import model_cards
b = 16
s = 512
n = 100
model_size = "350m"
config = model_cards[model_size]
mem_estimator = create_mem_estimator(b, s, n, config)

# two node test
sharding_strategy = {
    0: {},
    1: {
        0: {'shard': [0, 1], 'bits': ['8:tc', 16]},
        1: {'shard': [0, 1], 'bits': [16, '8:tc']},
    },
    2: {
        2: {'shard': [0, 1], 'bits': [16, 16]},
        3: {'shard': [0, 1], 'bits': ['8:tc', 16]},
        4: {'shard': [0, 1], 'bits': [16, 16]},
    },
    3: {
        5: {'shard': [0, 1], 'bits': [16, 8]},
        6: {'shard': [0, 1], 'bits': ['8:tc-li', 16]},
    },
    4: {
        7: {'shard': [0, 1], 'bits': [4, 16]},
        8: {'shard': [0], 'bits': [16]},
    },
    5: {
        8: {'shard': [1], 'bits': [16]},
        9: {'shard': [0,1], 'bits': [16, 2]},
        10: {'shard': [0,1], 'bits': [8, 16]},
        11: {'shard': [0,1], 'bits': [16, 16]},
    },
    6:{
        12: {'shard': [0,1], 'bits': [16, 16]},
        13: {'shard': [0,1], 'bits': [16, 16]},
        14: {'shard': [0,1], 'bits': [8, 16]},
        15: {'shard': [0,1], 'bits': [16, 16]},
        16: {'shard': [0,1], 'bits': [16, 16]},
        17: {'shard': [0,1], 'bits': [16, 8]},
    },
    7:{
        18: {'shard': [0,1], 'bits': [16, 16]},
        19: {'shard': [0,1], 'bits': [16, 16]},
        20: {'shard': [0,1], 'bits': [8, 16]},
        21: {'shard': [0,1], 'bits': [16, 16]},
        22: {'shard': [0,1], 'bits': [16, 16]}, 
        23: {'shard': [0,1], 'bits': [16, 16]},
    }
}

mem_estimator.calculate_maximum_mem_occupation_of_partition