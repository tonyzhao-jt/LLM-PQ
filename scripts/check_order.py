from utils import common_argparser, get_final_strat_file_name
import pickle
args = common_argparser()
model_name = args.model_name
model_size = args.model_size
device_info = args.device_info
file_name = get_final_strat_file_name(model_name, model_size, device_info)
folder = args.store_folder
abs_file_name = folder + '/' + file_name
with open(abs_file_name, 'rb') as f:
    sols = pickle.load(f)

test_method = args.test_method
sol = sols[test_method]
D = sol['D']
print(D)
print(device_info)