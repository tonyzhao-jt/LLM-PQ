export CUDA_VISIBLE_DEVICES=0,1
# load model weight
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 6666 main.py \
#     --model_name bloom --model_size 560m\
#     --sample-run --bs_token 1 --bitwidth 16 \
#     2>&1 | tee "text_res/sample_bloom_1.1_run_16.txt"
# pkill torchrun
# pkill python3

torchrun --nnodes=1 --nproc_per_node=2 --master_port 6666 main.py \
    --model_name opt --model_size 125m\
    --sample-run --bs_token 1 --bitwidth 16 --prompt_length 128 \
    2>&1 | tee "text_res/sample_bloom_1.1_run_16.txt"
pkill torchrun
pkill python3
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


