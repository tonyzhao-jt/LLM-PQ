export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/scripts/fakeCalib"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export OMP_NUM_THREADS=20
# CPU Not Enough
export LOAD_IN_NP="1"

STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2_gamma_0.6"

storage_folder="MULTI_NODE_PERF_RESULT"
mkdir $storage_folder

method=llm_pq # from adabits, uniform, pipeedge
# MASTER_ADDR=***REMOVED***
# method='llm_pq' # from adabits, llm_pq, uniform, pipeedge
MASTER_ADDR=***REMOVED***
MASTER_PORT=1234

SAMPLE_LOWER=0.2
# SAMPLE_LOWER=0.4
# SAMPLE_LOWER=0.6
# SAMPLE_LOWER=0.8
WORKLOAD_STRING="--workload-test --workload-nums 10 --sampler-lower ${SAMPLE_LOWER}"
# llm_pq-algo-check --file_path $ROOT_DIR/scripts/part_strategy/sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2group_1.pkl

llm_pq-dist --master_addr $MASTER_ADDR --master_port $MASTER_PORT\
    --method $method \
    --strat_file_name $STRAT_FILE_NAME $WORKLOAD_STRING \
        2>&1 | tee "./${storage_folder}/${method}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}_${SAMPLE_LOWER}.txt"
    pkill torchrun
    pkill python3

