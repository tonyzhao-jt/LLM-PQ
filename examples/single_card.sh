# STRAT_FILE_NAME="sols_opt_13b_Tesla_V100-SXM2-32GB_1"
STRAT_FILE_NAME="sols_opt_13b_NVIDIA_A100-SXM4-40GB_1"
methods=('adabits' 'shaq' 'uniform' 'pipeedge')
STRAT_FILE_NAME="sols_opt_13b_Tesla_V100-SXM2-32GB_1"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export LOAD_IN_NP="1"
mkdir SINGLE_NODE_PERF_RESULT

for method in "${methods[@]}"; do
    echo "Running method: $method"
    shaq-sole \
        --method $method \
        --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./SINGLE_NODE_PERF_RESULT/${method}_{$STRAT_FILE_NAME}.txt"
    pkill python3
done