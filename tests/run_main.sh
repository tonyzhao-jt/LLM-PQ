torchrun --nnodes=1 --nproc_per_node=4 main.py



torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 7000 --node_rank=0 main.py
torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 7000 --node_rank=1 main.py