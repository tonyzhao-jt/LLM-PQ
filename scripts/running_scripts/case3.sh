
export TRANSFORMERS_CACHE='/mnt/bn/zjtnaslq/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export OMP_NUM_THREADS=20
# CPU Not Enough
export LOAD_IN_NP="1"

STRAT_FILE_NAME="sols_opt_30b_Tesla_T4_3_Tesla_V100-SXM2-32GB_1group1"

storage_folder="MULTI_NODE_PERF_RESULT"
mkdir $storage_folder

# case 3 result has order problem
# method='llm_pq' # from adabits, llm_pq, uniform, pipeedge
# MASTER_ADDR=<>
method='pipeedge' # from adabits, llm_pq, uniform, pipeedge
MASTER_ADDR=<>
MASTER_PORT=1234

llm_pq-dist --master_addr $MASTER_ADDR --master_port $MASTER_PORT\
    --method $method \
    --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./${storage_folder}/${method}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}.txt"
    pkill torchrun
    pkill python3

