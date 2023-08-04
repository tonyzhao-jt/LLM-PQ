export CUDA_VISIBLE_DEVICES=1,2
export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export OMP_NUM_THREADS=20
# CPU Not Enough
export LOAD_IN_NP="1"

# STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2group_shaqh"
# STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2group_2"
STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2group_1"

SAMPLE_LOWER_RATIO=0.4
WORKLOAD_STRING="--workload-test --workload-nums 10 --sampler-lower $SAMPLE_LOWER_RATIO"

storage_folder="MULTI_NODE_PERF_RESULT"
mkdir $storage_folder

method=pipeedge # from adabits, uniform, pipeedge
# MASTER_ADDR=***REMOVED***
# method='shaq' # from adabits, shaq, uniform, pipeedge
MASTER_ADDR=***REMOVED***
MASTER_PORT=1444

# shaq-algo-check --file_path $ROOT_DIR/scripts/part_strategy/sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2group_shaqh.pkl

shaq-dist --master_addr $MASTER_ADDR --master_port $MASTER_PORT\
    --method $method \
    $WORKLOAD_STRING \
    --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./${storage_folder}/${method}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}_${SAMPLE_LOWER_RATIO}.txt"
    # pkill torchrun
    # pkill python3

