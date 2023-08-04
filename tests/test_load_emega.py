import pickle
import os 
ROOT_DIR = os.environ.get('ROOT_DIR')
omega_file_path = f"{ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl"
hess_file_path = f"{ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_hess_ind.pkl"
# load
with open(omega_file_path, 'rb') as f:
    omega = pickle.load(f)
with open(hess_file_path, 'rb') as f:
    hess = pickle.load(f)

import pdb; pdb.set_trace()