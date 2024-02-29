#!/bin/bash
# apt install openmpi-bin
HEAD_IP=$(hostname -i)
all_hosts=$HEAD_IP
N_CORES_PER_GPU=4

PYTHON_EXEC=$(which python3)
PYTHON_SCRIPT=flexgen.dist_flex_opt

ps ax | awk '/python3.*dist_flex_opt\.py/ {print $1}' | xargs kill

set -x
ENV="CUDA_VISIBLE_DEVICES=1,2"
HOSTS=(net-g15 ***REMOVED***)
PER_NODE_GPUS=(1 1)
# process
GPU_ALL=0
for gpus in "${PER_NODE_GPUS[@]}"
do
  GPU_ALL=$((GPU_ALL + gpus))
done
echo $GPU_ALL
# declare the array
npernode="--npernode $(printf "%s," "${PER_NODE_GPUS[@]}")"
npernode=${npernode%,}
mapping="${npernode} \
    -bind-to socket"
echo $mapping

tmp_base="--mca orte_tmpdir_base /opt/tiger/openmpi_tmp"

# specify channels
# exclude communication
# specify socket IFNAME
network_config="--mca btl ^openib,smcuda \
          --mca btl_tcp_if_exclude docker0,lo \
          --mca oob_tcp_if_exclude lo,docker0"
ALLOW_ROOT=""
if [[ "$USER_ENV" == "root" ]]; then
  ALLOW_ROOT="--allow-run-as-root"
fi

mpirun \
  -x ${ENV} \
  ${ALLOW_ROOT} \
  ${network_config} \
  ${tmp_base} \
  ${mapping} \
  -n $GPU_ALL --oversubscribe -H $all_hosts \
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $HEAD_IP \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-1.3b \
    --percent 0 100 0 100 0 100 \
    --comm-device cpu \
    --async-comm \
    --path _DUMMY_ \
    --prompt-len 512 --gen-len 80 --gpu-batch-size 8 --num-gpu-batches 1 \
    --pin-weight 0 