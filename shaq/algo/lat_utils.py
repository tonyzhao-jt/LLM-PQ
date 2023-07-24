import math
from ..partitioner.helper import (
    lat_prediction,
)
from .utils import get_comm_payload_size
def calculate_max_stage_lat(D, use_plan, \
                                       cost_model_pack, b, s=1, i=1, use_profiler_prediction=False, comm_size=0):
    lat_cost_model, comm_cost_model = cost_model_pack

    minmax_lat = 0
    stage_sum = 0

    stage_lat_list = []
    comm_lat_list = []
    for device_rank, shard_strategy in use_plan.items():
        stage_lat = 0
        D_name = D[device_rank]
        for layer_idx, layer_spec in shard_strategy.items():
            shard = layer_spec['shard']
            bit = layer_spec['bits']
            atten_bit, ffn_bit = bit
            stage_lat += lat_prediction(lat_cost_model, D_name, b, s, i, atten_bit, ffn_bit, use_profiler_prediction=use_profiler_prediction)
        # next stage
        next_stage = (device_rank + 1) % len(D)
        t_comm = comm_cost_model.predict_comm_time(device_rank, next_stage, comm_size)
        # minmax throughput
        minmax_lat = max(minmax_lat, stage_lat, t_comm)
        stage_sum += stage_lat
        stage_lat_list.append(stage_lat)
        comm_lat_list.append(t_comm)
    
    return (minmax_lat, stage_sum), (stage_lat_list, comm_lat_list)

def run_simu(gen_config, sol, lat_cost_model, comm_cost_model, use_profiler_prediction, mu_n, comm_multiplier):
    D = sol['D']
    use_plan = sol['use_plan']
    prefill_bz = sol['prefill_bz']
    bz_decode_max = sol['bz_decode_max']
    maps = sol['maps']
    if maps is not None:
        comm_cost_model.set_device_rank_map(maps)
    global_bz = gen_config.global_bz
    data_pack = (prefill_bz, bz_decode_max)
    cost_model_pack = (lat_cost_model, comm_cost_model)
    s = gen_config.s
    n = gen_config.n

    comm_size_prefill, comm_size_decode = get_comm_payload_size(lat_cost_model, s, prefill_bz, bz_decode_max, comm_multiplier)
    # comm_size_prefill = lat_cost_model.h1 * s * prefill_bz * 2 / 1024 / 1024 * comm_multiplier
    # comm_size_decode = lat_cost_model.h1 * 1 * bz_decode_max * 2 / 1024 / 1024 * comm_multiplier

    sol_name = sol['name']
    # if sol_name == 'shaq':
    #     import pdb; pdb.set_trace()
    # average throughput should equals to 
    (prefill_result, prefill_sum), (prefill_lat_list, prefill_comm_lat_list) = calculate_max_stage_lat(D, use_plan, \
                                                    cost_model_pack, prefill_bz, s, 0, use_profiler_prediction, comm_size_prefill)
    (decode_result, decode_sum), (decode_lat_list, decode_comm_lat_list) = calculate_max_stage_lat(D, use_plan, \
                                                    cost_model_pack, bz_decode_max, 1, s + int(mu_n / 2), use_profiler_prediction,  comm_size_decode)
    
    # print("Prefill cost stage", prefill_lat_list, prefill_comm_lat_list)
    # print("Decode cost stage", decode_lat_list, decode_comm_lat_list)
    print("Prefill cost stage", prefill_lat_list)
    print("Decode cost stage", decode_lat_list)
    prefill_micro_bs_num = math.ceil(global_bz / prefill_bz)
    decode_micro_bs_num = math.ceil(global_bz / bz_decode_max)
    prefill_time = prefill_sum + prefill_result * (prefill_micro_bs_num - 1)
    decode_time = decode_sum + decode_result * (decode_micro_bs_num - 1) * (mu_n - 1)
    # latency equals
    e2e_lat = prefill_time + decode_time

    print("Prefill Time {:.2f}ms, Decode Time {:.2f}ms, E2E Latency {:.2f}ms".format(prefill_time, decode_time, e2e_lat))
    # remove maps
    if maps is not None:
        comm_cost_model.clear_device_rank_map() 
    return e2e_lat