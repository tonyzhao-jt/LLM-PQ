export TRANSFORMERS_CACHE='/mnt/bn/zjtnaslq/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export OMP_NUM_THREADS=20
# CPU Not Enough
# we use perf mode here due to the device shortage of our emulation cluster we cannot run a bloom weight conversion and store its np weight
export LOAD_IN_NP="0" # we need perf mode now

STRAT_FILE_NAME="sols_bloom_176b_Tesla_V100-SXM2-32GB_4_NVIDIA_A100-SXM4-80GB_2"

storage_folder="MULTI_NODE_PERF_RESULT"
mkdir $storage_folder

method=adabits # from adabits, uniform, pipeedge
# MASTER_ADDR=***REMOVED***
# method='llm_pq' # from adabits, llm_pq, uniform, pipeedge
MASTER_ADDR=10.130.24.87
MASTER_PORT=1234
NNODES=2
NPROC_PER_NODE=2
RANK=1

llm_pq-dist --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=${RANK} \
    --model_name  bloom --model_size 176b --no_auto \
    --method $method --perf-mode \
    --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./${storage_folder}/${method}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}.txt"
    pkill torchrun
    pkill python3

