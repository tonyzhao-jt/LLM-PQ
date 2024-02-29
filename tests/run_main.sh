
export TP_SOCKET_IFNAME=enp225s0
export NCCL_SOCKET_IFNAME=enp225s0
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export TP_SOCKET_IFNAME=enp94s0
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export GLOO_SOCKET_IFNAME=enp94s0

# p2p
torchrun --nnodes=2 --nproc_per_node=4 --master_addr ***REMOVED*** --master_port 6666 --node_rank=0 main_p2p.py
torchrun --nnodes=2 --nproc_per_node=4 --master_addr ***REMOVED*** --master_port 6666 --node_rank=1 main_p2p.py

# test scripts for functionality
torchrun --nnodes=1 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 test_single_p2p.py --model_name bloom --model_size "560m"
torchrun --nnodes=1 --nproc_per_node=4 --master_port 6666 test_single_p2p.py --model_name opt --model_size "350m"
torchrun --nnodes=1 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 test_single.py

torchrun --nnodes=1 --nproc_per_node=4 --master_port 5555 test_single_p2p_tp.py --model_name opt --model_size "350m"

# rpc
torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 --node_rank=0 main.py
torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 6666 --node_rank=1 main.py

# RPC test dist
torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 7777 --node_rank=0 test_dist.py
torchrun --nnodes=2 --nproc_per_node=4 --master_addr net-g14 --master_port 7777 --node_rank=1 test_dist.py

# available
torchrun --nnodes=1 --nproc_per_node=4 --master_port 9999 test_rpc_tp.py --model_name opt --model_size "350m"
torchrun --nnodes=1 --nproc_per_node=4 --master_port 9999 test_rpc_tp.py --model_name bloom --model_size "560m"
torchrun --nnodes=1 --nproc_per_node=4 --master_port 6666 test_single_p2p.py --model_name bloom --model_size "560m"
torchrun --nnodes=1 --nproc_per_node=4 --master_port 6666 test_single_p2p.py --model_name opt --model_size "6.7b" --nccl

# p2p test dist
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 --master_addr ***REMOVED*** --master_port 6666 main_p2p.py --model_name opt --model_size "350m"
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=2 --master_addr ***REMOVED*** --master_port 6666 main_p2p.py --model_name opt --model_size "350m"


