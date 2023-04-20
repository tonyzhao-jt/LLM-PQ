torchrun --nnodes=1 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 main.py 

torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 --node_rank=0 main.py
torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 --node_rank=1 main.py

export TP_SOCKET_IFNAME=enp225s0
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 --node_rank=0 test_dist.py

export TP_SOCKET_IFNAME=enp94s0
torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 --node_rank=1 test_dist.py