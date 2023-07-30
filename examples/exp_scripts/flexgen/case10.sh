NUM_GPU_BATCHES = 4 
GPU_BATCH_SIZE=8 # 32/4
PARTITION_NUMS=4 # 4 gpus
# no nvme for the device
# for 4V100
# python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 160 \
#                      --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 30 --nvme-mem 0 \
#                       --num-gpu-batches 4 --partition_nums 4 

# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=0.77272193, w_cpu_percent=0.22727807, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 77 23 0 100 0 100

# python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 160 \
#                      --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 30 --nvme-mem 0 \
#                       --num-gpu-batches 4 --partition_nums 4 --compress-w

# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 100 0 100 0

#!/bin/bash
N_NODES=1
N_PROCES=4
ALL_GPUS=4
MASTER_ADDR=***REMOVED***
MASTER_PORT=1234
RANK=0

set -x
export OMP_NUM_THREADS=6
torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-66b \
    --prompt-len 512 --gen-len 80 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 77 23 0 100 0 100 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 

torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-66b \
    --prompt-len 512 --gen-len 80 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 100 0 100 0 100 0 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 \
    --compress-w

# 200
# python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 200 \
#                      --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 30 --nvme-mem 0 \
#                       --num-gpu-batches 4 --partition_nums 4 
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=0.77272193, w_cpu_percent=0.22727807, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 77 23 0 100 0 100

# python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 200 \
#                      --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 30 --nvme-mem 0 \
#                       --num-gpu-batches 4 --partition_nums 4 --compress-w
# # 100 0 100 0 100 0

torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-66b \
    --prompt-len 512 --gen-len 100 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 77 23 0 100 0 100 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 

torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-66b \
    --prompt-len 512 --gen-len 100 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 100 0 100 0 100 0 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 \
    --compress-w