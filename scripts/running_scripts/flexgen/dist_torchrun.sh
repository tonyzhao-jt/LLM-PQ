#!/bin/bash
N_NODES=1
N_PROCES=2
MASTER_ADDR=net-g15
MASTER_PORT=1234

set -x
export OMP_NUM_THREADS=6
torchrun --nnode $N_NODES --nproc_per_node=$N_PROCES --master_addr $MASTER_ADDR --master_port $MASTER_PORT dist_flex_opt.py \
    --model facebook/opt-1.3b \
    --prompt-len 512 --gen-len 80 --gpu-batch-size 8 --num-gpu-batches 1 \
    --percent 100 0 100 0 100 0 \
    --num-inner-iterations 2 \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --pin-weight 0 \
    --overlap False
