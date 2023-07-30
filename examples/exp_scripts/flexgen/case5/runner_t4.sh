#!/bin/bash
N_NODES=2
N_PROCES=4
ALL_GPUS=6
MASTER_ADDR=***REMOVED***
MASTER_PORT=1234
RANK=0

set -x
export OMP_NUM_THREADS=6
# torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
#     --model facebook/opt-66b \
#     --prompt-len 512 --gen-len 80 --gpu-batch-size 6 --num-gpu-batches 1 \
#     --percent 34 66 0 100 0 100 \
#     --num-inner-iterations $ALL_GPUS \
#     --comm-device cpu \
#     --async-comm \
#     --path _DUMMY_ \
#     --pin-weight 0 


# torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
#     --model facebook/opt-66b \
#     --prompt-len 512 --gen-len 80 --gpu-batch-size 6 --num-gpu-batches 1 \
#     --percent 100 0 54 46 0 100 \
#     --num-inner-iterations $ALL_GPUS \
#     --comm-device cpu \
#     --async-comm \
#     --path _DUMMY_ \
#     --pin-weight 0 \
#     --compress-w


torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-66b \
    --prompt-len 512 --gen-len 100 --gpu-batch-size 6 --num-gpu-batches 1 \
    --percent 33 66 0 100 0 100 \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 


torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=$RANK dist_flex_opt.py \
    --model facebook/opt-66b \
    --prompt-len 512 --gen-len 100 --gpu-batch-size 6 --num-gpu-batches 1 \
    --percent 100 0 52 48 0 100  \
    --num-inner-iterations $ALL_GPUS \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 \
    --compress-w