import pickle
import argparse
def check_strat_file():
    # allow user input the file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    # test_method, choose from llm_pq, adabits, pipeedge, adabits
    parser.add_argument('--test_method', type=str, default='llm_pq', help='test method')
    args = parser.parse_args()
    file_path = args.file_path
    # load sols
    with open(file_path, 'rb') as f:
        sols = pickle.load(f)

    test_method = args.test_method
    sol = sols[test_method]
    print("Hybrid batching plan:")
    print(sol['prefill_bz'], sol['bz_decode_max'], sol['bz_decode_bss'])
    print("Device:")
    print(sol['D'])
    print("Use plan:")
    print(sol['use_plan'])
    print("Partition Result:")
    print(sol['plan']['partition_result'])
    print("Token gen num:")
    print(sols['mu_n'])
    bit_assignment = sol['plan']['bit_assignment']
    # check how many bitwidth is used in the plan
    bitwidths = set()
    for bit in bit_assignment.values():
        bitwidths.add(bit)
    print(f"Bitwidths used: {bitwidths}")


def hamming_distance(list1, list2):
    """
    Calculates the Hamming distance between two lists of categorical values.
    """
    if len(list1) != len(list2):
        raise ValueError("The two lists must be of equal length.")
    
    distance = 0
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            distance += 1
    
    return distance

def compare_bitwidth_of_two_strat():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path_1', type=str, required=True)
    parser.add_argument('--file_path_2', type=str, required=True)
    args = parser.parse_args()
    test_method = 'llm_pq'
    file_path_1 = args.file_path_1
    file_path_2 = args.file_path_2
    # load sols
    with open(file_path_1, 'rb') as f:
        sols_1 = pickle.load(f)
    with open(file_path_2, 'rb') as f:
        sols_2 = pickle.load(f)
    
    sol_1 = sols_1[test_method]
    sol_2 = sols_2[test_method]
    # use_plan
    use_plan_1 = sol_1['use_plan']
    use_plan_2 = sol_2['use_plan']
    bit_assignment_1 = sol_1['plan']['bit_assignment']
    bit_assignment_2 = sol_2['plan']['bit_assignment']
    # print(sol_1)
    # # compare the bitwidth difference
    values_bits_1 = list(bit_assignment_1.values())
    values_bits_2 = list(bit_assignment_2.values())
    hamming_val = hamming_distance(values_bits_1, values_bits_2)
    has_diff = False
    for i in range(len(values_bits_1)):
        if values_bits_1[i] != values_bits_2[i]:
            print(f"layer {i}: {values_bits_1[i]} vs {values_bits_2[i]}")
            has_diff = True
    if not has_diff:
        print("No difference")
    else:
        print(f"Hamming distance: {hamming_val}")

