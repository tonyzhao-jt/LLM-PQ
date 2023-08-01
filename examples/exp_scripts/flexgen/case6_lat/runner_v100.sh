#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
N_NODES=2
N_PROCES=2
ALL_GPUS=4
MASTER_ADDR=***REMOVED***
MASTER_PORT=1234
RANK=0

set -x
export OMP_NUM_THREADS=6
# torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
#     --model facebook/opt-66b \
#     --prompt-len 128 --gen-len 160 --gpu-batch-size 8 --num-gpu-batches 1 \
#     --percent 81 19 0 100 0 100 \
#     --num-inner-iterations $ALL_GPUS \
#     --comm-device cpu \
#     --async-comm \
#     --path _DUMMY_ \
#     --pin-weight 0 

# torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
#     --model facebook/opt-66b \
#     --prompt-len 128 --gen-len 160 --gpu-batch-size 8 --num-gpu-batches 1 \
#     --percent 100 0 100 0 100 0 \
#     --num-inner-iterations $ALL_GPUS \
#     --comm-device cpu \
#     --async-comm \
#     --path _DUMMY_ \
#     --pin-weight 0 \
#     --compress-w


torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-66b \
    --prompt-len 128 --gen-len 200 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 81 19 0 100 0 100 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 

torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-66b \
    --prompt-len 128 --gen-len 200 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 100 0 100 0 100 0 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 \
    --compress-w