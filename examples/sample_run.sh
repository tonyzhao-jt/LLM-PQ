export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_CACHE='/data/llms/'

# CPU Not Enough
export LOAD_IN_NP="1"
python3 weight_convert_numpy.py --model-size 125m # run this in other node has enough cpu memory, then scp to the target device

NUM_SHARDS=2
torchrun --nnodes=1 --nproc_per_node=$NUM_SHARDS --master_port 6666 main.py \
    --model_name opt --model_size 125m\
    --sample-run --bs_token 10 --bitwidth 16 --prompt_length 128 --num-shards $NUM_SHARDS \
    2>&1 | tee "text_res/sample_bloom_1.1_run_16.txt"
pkill torchrun
pkill python3

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


