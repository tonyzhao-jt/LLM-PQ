# read partition from file
import pickle
pipeline_strategy_result_qpipe = "pipeline_strategy_result_qpipe.pkl"
pipeline_strategy_result_qpipe = f'/workspace/qpipe/scripts/baseline_result/{pipeline_strategy_result_qpipe}'
qpipe_partition_strategies = pickle.load(open(pipeline_strategy_result_qpipe, "rb"))
print(qpipe_partition_strategies[0])