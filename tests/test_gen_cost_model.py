'''
    Test the generation of the latency cost model
'''

from qpipe.cost_model import LatCostModel
from qpipe.partitioner import gen_config
from qpipe.partitioner.helper import (
    lat_prediction
)
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
h1, h2 = 7168, 28672
lat_cost_model.h1 = h1 
lat_cost_model.h2 = h2
# groudh truth
# predict(self, device_name, shard, b, s, i, h1, h2, bit)
# D_name = 'Tesla_V100-SXM2-32GB'
D_name = 'NVIDIA_A100-SXM4-40GB'
b = 1
# prefill
i = 0 
atten_bit = ffn_bit = 16 
predict_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=False)
groud_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=True)
print(predict_lat, groud_lat)
atten_bit = ffn_bit = '8:tc-li'
predict_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=False)
groud_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=True)
print(predict_lat, groud_lat)
atten_bit = ffn_bit = 4
predict_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=False)
groud_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=True)
print(predict_lat, groud_lat)
# decodde
i = s + n 
s = 1
atten_bit = ffn_bit = 16 
predict_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=False)
groud_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=True)
print(predict_lat, groud_lat)
# predict non exists result
atten_bit = ffn_bit = '8:tc-li'
predict_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=False)
groud_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=True)
print(predict_lat, groud_lat)
atten_bit = ffn_bit = 4
predict_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=False)
groud_lat = lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=True)
print(predict_lat, groud_lat)