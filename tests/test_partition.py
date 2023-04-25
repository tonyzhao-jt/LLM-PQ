# read partition from file
import pickle, json
pipeline_strategy_result_qpipe = "qpipe_Tesla_V100-SXM2-32GB_4_NVIDIA_A100-SXM4-40GB_4_final_strategy.pkl"
pipeline_strategy_result_qpipe = f'/workspace/qpipe/scripts/part_strategy/{pipeline_strategy_result_qpipe}'
qpipe_partition_strategies = pickle.load(open(pipeline_strategy_result_qpipe, "rb"))
import pdb; pdb.set_trace()
# print it human readable
print(json.dumps(qpipe_partition_strategies, indent=4))

