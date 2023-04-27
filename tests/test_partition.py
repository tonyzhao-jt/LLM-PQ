# read partition from file
import pickle, json
model_size = "30b"
method = 'qpipe'
device_info = 'Tesla_T4_4_Tesla_V100-SXM2-32GB_1'
pipeline_strategy_result_qpipe = f"{method}_{model_size}_{device_info}_final_strategy.pkl"
pipeline_strategy_result_qpipe = f'../scripts/part_strategy/{pipeline_strategy_result_qpipe}'
qpipe_partition_strategies = pickle.load(open(pipeline_strategy_result_qpipe, "rb"))
# print it human readable
print(json.dumps(qpipe_partition_strategies, indent=4))

