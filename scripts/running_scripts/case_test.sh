method='adabits' # from adabits, llm_pq, uniform, pipeedge
STRAT_FILE_NAME="sols_opt_125m_Tesla_V100-SXM2-32GB_1"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/scripts/fakeCalib"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
# CPU Not Enough
export LOAD_IN_NP="1"
mkdir SINGLE_NODE_PERF_RESULT


llm_pq-dist --master_port 1454 \
    --method $method \
    --strat_file_name $STRAT_FILE_NAME \
    2>&1 | tee "./SINGLE_NODE_PERF_RESULT/${method}_{$STRAT_FILE_NAME}.txt"
pkill torchrun
pkill python3
