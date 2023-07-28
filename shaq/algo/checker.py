import pickle
import argparse
def check_strat_file():
    # allow user input the file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    # test_method, choose from shaq, adabits, pipeedge, adabits
    parser.add_argument('--test_method', type=str, default='shaq', help='test method')
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

def compare_bitwidth_of_two_strat():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path_1', type=str, required=True)
    parser.add_argument('--file_path_2', type=str, required=True)
    args = parser.parse_args()
    test_method = 'shaq'
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
    # compare the bitwidth difference
    for rank, val in use_plan_1.items():
        use_plan_2_val = use_plan_2[rank]
        bit_1 = val['bitwidth']
        bit_2 = use_plan_2_val['bitwidth']
        for idx, bit in enumerate(bit_1):
            if bit != bit_2[idx]:
                print(f"Rank {rank} layer {idx}: {bit} vs {bit_2[idx]}")
            # break
            break 