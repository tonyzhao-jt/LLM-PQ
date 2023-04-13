from qpipe.cost_model import fit_cost_model, LatCostModel

profiled_result_folder = '/workspace/qpipe/scripts/lat_profiled_result/'
cost_model_store_path = '/workspace/qpipe/scripts/lat_cost_model'
target_device = 'Tesla_V100-SXM2-32GB'
target_device = 'NVIDIA_A100-SXM4-40GB'
fit_cost_model(profiled_result_folder, cost_model_store_path, target_device)