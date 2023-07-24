
export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export OMP_NUM_THREADS=20
# CPU Not Enough
export LOAD_IN_NP="1"

method='adaqpipe' # from adabits, shaq, uniform, pipeedge
STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2"

storage_folder="MULTI_NODE_PERF_RESULT"
mkdir $storage_folder
# A100-Single node
MASTER_ADDR=net-g15
MASTER_PORT=1234

shaq-dist --master_addr $MASTER_ADDR --master_port $MASTER_PORT\
    --method $method --model_size 66b \
    --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./${storage_folder}/${method}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}.txt"


