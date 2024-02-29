#!/bin/bash
N_NODES=2
N_PROCES=3
ALL_GPUS=4
MASTER_ADDR=***REMOVED***
MASTER_PORT=1234
RANK=0

set -x
export OMP_NUM_THREADS=6
# torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
#     --model facebook/opt-30b \
#     --prompt-len 512 --gen-len 80 --gpu-batch-size 8 --num-gpu-batches 1 \
#     --percent 83 17 0 100 0 100 \
#     --num-inner-iterations $ALL_GPUS \
#     --comm-device cpu \
#     --async-comm \
#     --path _DUMMY_ \
#     --pin-weight 0 



# torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
#     --model facebook/opt-30b \
#     --prompt-len 512 --gen-len 80 --gpu-batch-size 8 --num-gpu-batches 1 \
#     --percent 100 0 100 0 0 100 \
#     --num-inner-iterations $ALL_GPUS \
#     --comm-device cpu \
#     --async-comm \
#     --path _DUMMY_ \
#     --pin-weight 0 \
#     --compress-w


# 100

torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-30b \
    --prompt-len 512 --gen-len 100 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 63 37 0 100 0 100 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 



torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-30b \
    --prompt-len 512 --gen-len 100 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 100 0 97 3 0 100 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 \
    --compress-w