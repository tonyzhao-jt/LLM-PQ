export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export OMP_NUM_THREADS=20
# CPU Not Enough
# we use perf mode here due to the device shortage of our emulation cluster we cannot run a bloom weight conversion and store its np weight
export LOAD_IN_NP="0" # we need perf mode now

# STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_4_NVIDIA_A100-SXM4-40GB_4noq_1024_32"
# STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_4_NVIDIA_A100-SXM4-40GB_4noq_512_32"
STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_4_NVIDIA_A100-SXM4-40GB_4noq_128_32"
storage_folder="MULTI_NODE_PERF_RESULT"
mkdir $storage_folder

method=shaq # from adabits, uniform, pipeedge
# MASTER_ADDR=net-g13
# method='shaq' # from adabits, shaq, uniform, pipeedge
MASTER_ADDR=net-g13
MASTER_PORT=1234

shaq-dist --master_addr $MASTER_ADDR --master_port $MASTER_PORT\
    --method $method --perf-mode \
    --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./${storage_folder}/${method}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}.txt"
    pkill torchrun
    pkill python3

