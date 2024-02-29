def convert_D_to_ranked_device(D, device_name=None):
    rank = 0
    device_rank_tmp = 0
    device_rank_list = {}
    for device_rank, device_name_ in D.items():
        if device_name_ not in device_rank_list:
            device_rank_list[device_name_] = 1
            if device_name is not None and device_name == device_name_:
                rank = device_rank_tmp
            device_rank_tmp += 1
        else:
            device_rank_list[device_name_] += 1
    return device_rank_list, rank

