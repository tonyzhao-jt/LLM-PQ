def partition_a_into_b_bins(a, b):
    remainders = a % b
    ideal_allocation = a // b
    allocation = []
    for i in range(b):
        allocation.append(ideal_allocation) 
    for i in range(remainders):
        allocation[i] += 1 
    return allocation


def get_default_decode_bz(global_bz, num_device_all):
    bz_decode_max = max(partition_a_into_b_bins(global_bz, num_device_all))
    return bz_decode_max