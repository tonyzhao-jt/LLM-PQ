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
