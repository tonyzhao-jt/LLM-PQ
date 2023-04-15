from qpipe.cost_model import fit_cost_model, LatCostModel

profiled_result_folder = '/workspace/qpipe/scripts/lat_profiled_result/'
cost_model_store_path = '/workspace/qpipe/scripts/lat_cost_model'
lat_cost_model = LatCostModel(cost_model_store_path, ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB'])
lat_cost_model.register_hyper_params(16, 612, 12288, 49152)
lat_cost_model.update_profiled_result('/workspace/qpipe/scripts')
# res = lat_cost_model.predict_with_profiled('NVIDIA_A100-SXM4-40GB', 0, 8)
res = lat_cost_model.predict_with_profiled('Tesla_V100-SXM2-32GB', 0, '8-tc')
print(res)