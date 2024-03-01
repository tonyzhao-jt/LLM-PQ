import sys
import subprocess
import os 
import pkg_resources
import pickle
import socket
from ..utils import get_device_name_and_mem, convert_D_to_ranked_device
def run_dist():
    # Get the arguments passed to the llm_pq-dist command
    args = sys.argv[1:]

    # extract and insert user-specified arguments before script path
    idx_of_torchrun_args = 0
    has_strat_file = False
    master_addr = None
    master_port = None
    no_auto = False
    for i, arg in enumerate(args):
        if arg.startswith("--nnodes=") or arg.startswith("--nproc_per_node=") or arg.startswith("--node_rank="):
            if i >= idx_of_torchrun_args:
                idx_of_torchrun_args = i + 1
        elif arg.startswith("--master_addr"):
            master_addr = args[i+1]
            if i >= idx_of_torchrun_args:
                idx_of_torchrun_args = i + 2
        elif arg.startswith("--master_port"):
            master_port = args[i+1]
            if i >= idx_of_torchrun_args:
                idx_of_torchrun_args = i + 2
        elif arg.startswith("--strat_file_name"):
            has_strat_file = True
            strat_file_name = args[i+1]
        elif arg.startswith("--method"):
            method_name = args[i+1]
        elif arg.startswith("--no_auto"):
            no_auto = True
    # if no auto, pop from the args
    if no_auto:
        args.pop(args.index("--no_auto"))
    # check whether has the strat file, if has, then reorgnize the torch run args
    if has_strat_file and not no_auto:
        print("Reconstructed torchrun args:")
        sol_file = f"{strat_file_name}.pkl"
        root_dir = os.environ['ROOT_DIR']
        strat_folder = f'{root_dir}/scripts/part_strategy'
        sols_path = f'{strat_folder}/{sol_file}'
        sols = pickle.load(open(sols_path, "rb"))
        sol = sols[method_name]
        D = sol['D']
        device_name, _, _ = get_device_name_and_mem()
        # ref
        # {0: 'Tesla_T4', 1: 'Tesla_T4', 2: 'Tesla_T4', 3: 'Tesla_V100-SXM2-32GB'}
        # get index order of devices
        if 'A800' in device_name:
            device_name = "NVIDIA_A100-SXM4-80GB" # TODO: later fix the name
        device_rank_list, rank = convert_D_to_ranked_device(D, device_name=device_name)
        nnodes = len(device_rank_list)
        # current device index
        assert device_name in device_rank_list, "Run on the wrong device, please check the device"
        n_procs = device_rank_list[device_name]
        # reconstruct --nnodes= --nproc_per_node= --node_rank=
        new_args = []
        # Get the hostname 
        hostname = socket.gethostname()
        # Get the IP address 
        ip_address = socket.gethostbyname(hostname)
        if master_addr is None:
            master_addr = ip_address if ip_address != "127.0.0.1" else hostname
        # first add address and port
        new_args.append("--master_addr")
        new_args.append(master_addr)
        new_args.append("--master_port")
        new_args.append(master_port)
        # then add nnodes
        new_args.append("--nnodes")
        new_args.append(str(nnodes))
        # then add nproc_per_node
        new_args.append("--nproc_per_node")
        new_args.append(str(n_procs))
        # then add node_rank
        new_args.append("--node_rank")
        new_args.append(str(rank))
        args = new_args + args[idx_of_torchrun_args:]
        idx_of_torchrun_args = len(new_args)

    # Modify the arguments as needed
    args.insert(0, "torchrun")
    idx_of_torchrun_args += 1

    # append user-specified arguments before script path
    # script_path = os.path.join(os.getcwd(), "llm_pq", "dist_runtime", "entry.py")
    script_path = pkg_resources.resource_filename("llm_pq", "dist_runtime/entry.py")
    args.insert(idx_of_torchrun_args, script_path)
    print(args)
    # Call torch.distributed.launch.main() with the modified arguments
    subprocess.run(args)

def run_sole():
    args = sys.argv[1:]
    # Modify the arguments as needed
    args.insert(0, "python")
    # append user-specified arguments before script path
    # script_path = os.path.join(os.getcwd(), "llm_pq", "dist_runtime", "entry.py")
    script_path = pkg_resources.resource_filename("llm_pq", "dist_runtime/entry_sole.py")
    args.insert(1, script_path)
    # Call torch.distributed.launch.main() with the modified arguments
    subprocess.run(args)