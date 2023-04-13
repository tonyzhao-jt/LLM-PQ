from qpipe.cost_model import fit_cost_model, LatCostModel
profiled_result_folder = '/workspace/qpipe/scripts/lat_profiled_result/'
cost_model_store_path = '/workspace/qpipe/scripts/lat_cost_model'
device_names = ['Tesla_V100-SXM2-32GB', 'NVIDIA_A100-SXM4-40GB']
for target_device in device_names:
    fit_cost_model(profiled_result_folder, cost_model_store_path, target_device)


# veryfy the cost model
lat_cost_model = LatCostModel(cost_model_store_path, device_names)
# test a case
shard = 0
bit = 8
b = 16
h1 = 12288
h2 = 49152
s = 512
n = 100
i = s + n 
device = 'Tesla_V100-SXM2-32GB'

predicted_cost = lat_cost_model.predict(device, shard, b, i, h1, h2, bit)
print(f"Predicted cost: {predicted_cost.item()} ms")