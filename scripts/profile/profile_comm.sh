#!/bin/bash
NNODES=2
NUM_PROC=2
MASTER_IP=
MASTER_PORT=11234
export NCCL_SOCKET_IFNAME=

# get the IP address of the current node
IP=
if [ "$IP" = "$MASTER_IP" ]; then
    IS_MASTER=true
    echo "This is the master node"
else
    IS_MASTER=false
    echo "This is a worker node"
fi

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=20 NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_PROC \
    --rdzv_id=1234 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
    --rdzv_conf is_host=$IS_MASTER \
    profile_comm.py 

# profile_comm.py  --nccl