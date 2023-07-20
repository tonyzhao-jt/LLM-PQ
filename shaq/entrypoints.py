import sys
import subprocess
import os 
import pkg_resources
def run_dist():
    # Get the arguments passed to the shaq-dist command
    args = sys.argv[1:]

    # Modify the arguments as needed
    args.insert(0, "torchrun")
    # extract and insert user-specified arguments before script path
    idx_of_torchrun_args = 0
    for i, arg in enumerate(args):
        if arg.startswith("--nnodes=") or arg.startswith("--nproc_per_node="):
            if i > idx_of_torchrun_args:
                idx_of_torchrun_args = i + 1
        elif arg.startswith("--master_addr") or arg.startswith("--master_port"):
            if i > idx_of_torchrun_args:
                idx_of_torchrun_args = i + 2

    # append user-specified arguments before script path
    # script_path = os.path.join(os.getcwd(), "shaq", "dist_runtime", "entry.py")
    script_path = pkg_resources.resource_filename("shaq", "dist_runtime/entry.py")
    args.insert(idx_of_torchrun_args, script_path)
    print(args)
    # Call torch.distributed.launch.main() with the modified arguments
    subprocess.run(args)