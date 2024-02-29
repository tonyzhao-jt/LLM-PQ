method='uniform' # from adabits, llm_pq, uniform, pipeedge
STRAT_FILE_NAME="sols_bloom_176b_NVIDIA_A100-SXM4-80GB_4"
# STRAT_FILE_NAME="sols_opt_13b_NVIDIA_A100-SXM4-40GB_1"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE='/mnt/bn/zjtnaslq/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
# CPU Not Enough
export LOAD_IN_NP="0"
mkdir SINGLE_NODE_PERF_RESULT

llm_pq-dist --master_port 4321 \
    --method $method --perf-mode \
    --strat_file_name $STRAT_FILE_NAME \
    2>&1 | tee "./SINGLE_NODE_PERF_RESULT/${method}_{$STRAT_FILE_NAME}.txt"
pkill torchrun
pkill python3

