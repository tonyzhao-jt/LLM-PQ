
export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/scripts/fakeCalib"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export OMP_NUM_THREADS=20
# CPU Not Enough
export LOAD_IN_NP="1"

method='llm_pq' # from adabits, llm_pq, uniform, pipeedge
STRAT_FILE_NAME="sols_opt_30b_Tesla_P100-PCIE-12GB_3_Tesla_V100-SXM2-32GB_1"

storage_folder="MULTI_NODE_PERF_RESULT"
mkdir $storage_folder
# A100-Single node
# MASTER_ADDR=net-g1
MASTER_PORT=1234
NNODES=3

# master
# RANK=0
# NPROC_PER_NODE=2
# worker1
# RANK=1
# NPROC_PER_NODE=1
# # worker2
# RANK=2
# NPROC_PER_NODE=1

# master
MASTER_ADDR=***REMOVED***
# RANK=1
# NPROC_PER_NODE=2
# worker1
# RANK=2
# NPROC_PER_NODE=1
# # worker2
RANK=0
NPROC_PER_NODE=1

# specify no-auto here to avoid auto distributed generation for case
# two or more cluster with same cards, e.g. cluster 1. 2P100 cluster 2 1 P100
llm_pq-dist --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=${RANK} \
    --method $method --model_size 66b --no_auto --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./${storage_folder}/${method}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}.txt"


