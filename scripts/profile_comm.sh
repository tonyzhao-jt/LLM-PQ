
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_port 7000 --master_addr net-g14 profile_comm_cpu.py
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_port 7000 --master_addr net-g14 profile_comm_cpu.py


torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_port 7000 --master_addr net-g14 profile_comm_cpu.py
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_port 7000 --master_addr net-g14 profile_comm_cpu.py

torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_port 7000 --master_addr 10.139.117.21 profile_comm_cpu.py
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_port 7000 --master_addr 10.139.117.21 profile_comm_cpu.py