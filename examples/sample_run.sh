export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
# CPU Not Enough
export LOAD_IN_NP="1"
# python3 weight_convert_numpy.py --model-size 125m # run this in other node has enough cpu memory, then scp to the target device

MODEL_NAME='opt'
MODEL_SIZE="125m"
PROMPT_LEGNTH=128
# PROMPT_LEGNTH=1024
BS=8
BITWIDTH=16
NUM_TOKENS_TO_GENERATE=1
# NUM_TOKENS_TO_GENERATE=32
# NUM_TOKENS_TO_GENERATE=50
# NUM_TOKENS_TO_GENERATE=100
MAX_TOKENS_TO_GENERATE=100 # make sure it always larger than previous

NUM_SHARDS=1
# shaq direct run
shaq-dist --nnodes=1 --nproc_per_node=$NUM_SHARDS --master_port 1234 \
    --model_name $MODEL_NAME --model_size $MODEL_SIZE\
    --sample-run --bs_token $BS --bitwidth $BITWIDTH --prompt_length $PROMPT_LEGNTH --num-shards $NUM_SHARDS \
    --num_tokens_to_generate $NUM_TOKENS_TO_GENERATE --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    2>&1 | tee "text_res/sample_${MODEL_NAME}_${MODEL_SIZE}_run_${BITWIDTH}_${PROMPT_LEGNTH}_${NUM_TOKENS_TO_GENERATE}.txt"

# run by torch
# torchrun --nnodes=1 --nproc_per_node=$NUM_SHARDS --master_port 1234 main.py \
#     --model_name $MODEL_NAME --model_size $MODEL_SIZE\
#     --sample-run --bs_token $BS --bitwidth $BITWIDTH --prompt_length $PROMPT_LEGNTH --num-shards $NUM_SHARDS \
#     --num_tokens_to_generate $NUM_TOKENS_TO_GENERATE --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
#     2>&1 | tee "text_res/sample_${MODEL_NAME}_${MODEL_SIZE}_run_${BITWIDTH}_${PROMPT_LEGNTH}_${NUM_TOKENS_TO_GENERATE}.txt"
# pkill torchrun
# pkill python3

# CPU Enough
# export LOAD_IN_NP="0"
# python3 weight_convert.py --model-size 125m # sample run

# torchrun --nnodes=1 --nproc_per_node=1 --master_port 6666 main.py \
#     --model_name opt --model_size 125m\
#     --sample-run --bs_token 1 --bitwidth 16 --prompt_length 128 --num-shards 1 \
#     2>&1 | tee "text_res/sample_bloom_1.1_run_16.txt"
# pkill torchrun
# pkill python3

# load model weight
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 6666 main.py \
#     --model_name bloom --model_size 560m\
#     --sample-run --bs_token 1 --bitwidth 16 \
#     2>&1 | tee "text_res/sample_bloom_1.1_run_16.txt"
# pkill torchrun
# pkill python3

# int8
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 6666 main.py \
#     --model_name bloom --model_size 560m\
#     --sample-run --bs_token 8 --bitwidth "8:tc-li" \
#     2>&1 | tee "text_res/sample_bloom_1.1_run_8.txt"
# pkill torchrun
# pkill python3
# # you can compare it with the performance of the perf-mode
# # perf mode will greatly reduce the cpu usage for weight loading. 
# # it can provides perf result, bu cannot produce any useful tokens
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 6666 main.py \
#     --model_name bloom --model_size 560m\
#     --sample-run --bs_token 8 --bitwidth "8:tc-li" \
#     --perf-mode \
#     2>&1 | tee "text_res/sample_bloom_1.1_run_perf_8.txt"
# pkill torchrun
# pkill python3


