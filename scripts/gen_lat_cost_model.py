from qpipe.cost_model import LatCostModel
from qpipe.partitioner import gen_config
global_bz = gen_config.global_bz
micro_bz = gen_config.micro_bz
s = gen_config.s
n = gen_config.n

profiled_result_folder = '/workspace/qpipe/scripts/lat_profiled_result/'
device_names = ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
lat_cost_model = LatCostModel(device_names)
lat_cost_model.update_profiled_result(profiled_result_folder)
lat_cost_model.fit_regression_cost_model()
lat_cost_model.load_regression_cost_model()
lat_cost_model.verbose_regression_names()
# groudh truth
# predict(self, device_name, shard, b, s, i, h1, h2, bit)
# prefill
predict_lat = lat_cost_model.predict('Tesla_V100-SXM2-32GB', 1, 16, s, 0, 7168, 28672, 8)
groud_lat = lat_cost_model.fetch_lat('Tesla_V100-SXM2-32GB', 1, 16, s, 0, 7168, 28672, 8)
print(predict_lat, groud_lat)
predict_lat = lat_cost_model.predict('NVIDIA_A100-SXM4-40GB', 0, 4, s, 0, 7168, 28672, 8)
groud_lat = lat_cost_model.fetch_lat('NVIDIA_A100-SXM4-40GB', 0, 4, s, 0, 7168, 28672, 8)
print(predict_lat, groud_lat)
# decodde
predict_lat = lat_cost_model.predict('NVIDIA_A100-SXM4-40GB', 0, 4, 1, s+n, 7168, 28672, 8)
groud_lat = lat_cost_model.fetch_lat('NVIDIA_A100-SXM4-40GB', 0, 4, 1, s+n, 7168, 28672, 8)
print(predict_lat, groud_lat)
# predict non exists result
predict_lat = lat_cost_model.predict('Tesla_V100-SXM2-32GB', 0, 12, 1, s + n, 7168, 28672, 8)
groud_lat = lat_cost_model.fetch_lat('Tesla_V100-SXM2-32GB', 0, 12, 1, s + n, 7168, 28672, 8)
print(predict_lat, groud_lat)