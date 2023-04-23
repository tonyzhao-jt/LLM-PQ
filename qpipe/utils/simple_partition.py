def partition_a_into_b_bins(a, b):
    remainders = a % b
    ideal_allocation = a // b
    allocation = []
    for i in range(b):
        allocation.append(ideal_allocation) 
    for i in range(remainders):
        allocation[i] += 1 
    return allocation