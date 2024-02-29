torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_port 7000 --master_addr ***REMOVED*** profile_comm_cpu.py
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_port 7000 --master_addr ***REMOVED*** profile_comm_cpu.py

