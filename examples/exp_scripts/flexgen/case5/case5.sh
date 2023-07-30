# generate solution
# use uniform phase micro batch partition
# 32 / 6 cannot be well handled by the flexgen
# OPT66b: 64 / 6 = 10 .. 4
# 10 layers for each, 11 11 11 11, 10 10 for flexgen's algo.
# pipeline_stage_sizes = [config.num_hidden_layers // num_pipeline_stages
#                     + int(i < config.num_hidden_layers % num_pipeline_stages)
# as 32 mod 6 = 5 ... 2
# we take bs 36 result here, scale by 32/36
NUM_GPU_BATCHES = 6 
GPU_BATCH_SIZE=6 
PARTITION_NUMS=6 # 6 gpus
# no nvme for the device
# for T4
python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 6 --gpu-mem 15 --cpu-mem 41 --nvme-mem 0 \
                      --num-gpu-batches 6 --sl 11 --alpha-g 0.7 --alpha-c 0.7
# 0.8 causes OOM in our case.
# Policy(gpu_batch_size=6, num_gpu_batches=6, w_gpu_percent=0.33676269, w_cpu_percent=0.66323731, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 33 66 0 100 0 100

# for V100
python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 6 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 6 --sl 10

# Policy(gpu_batch_size=6, num_gpu_batches=6, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.77442776, cache_cpu_percent=0.22557224, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 77 23 0 100

# result
# CANNOT PROVIDE A VALID SOLUTION, ALWAYS FAILED

python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 6 --gpu-mem 15 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 6 --sl 11 --alpha-g 0.7 --compress-w
# Policy(gpu_batch_size=6, num_gpu_batches=6, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.53799254, cache_cpu_percent=0.46200746, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 54 46 0 100 

python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 6 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 6 --sl 10 --compress-w

# Policy(gpu_batch_size=6, num_gpu_batches=6, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 100 0 100 0
# result


# 100
NUM_GPU_BATCHES = 6 
GPU_BATCH_SIZE=6 
PARTITION_NUMS=6 # 6 gpus
# no nvme for the device
# for T4
python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 6 --gpu-mem 15 --cpu-mem 41 --nvme-mem 0 \
                      --num-gpu-batches 6 --sl 11 --alpha-g 0.7 --alpha-c 0.7
# 0.8 causes OOM in our case.
# Policy(gpu_batch_size=6, num_gpu_batches=6, w_gpu_percent=0.33676269, w_cpu_percent=0.66323731, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 33 66 0 100 0 100

# for V100
python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 6 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 6 --sl 10

# Policy(gpu_batch_size=6, num_gpu_batches=6, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.77442776, cache_cpu_percent=0.22557224, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 75 25 0 100

# result
# CANNOT PROVIDE A VALID SOLUTION, ALWAYS FAILED
python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 6 --gpu-mem 15 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 6 --sl 11 --alpha-g 0.7 --compress-w
# Policy(gpu_batch_size=6, num_gpu_batches=6, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.53799254, cache_cpu_percent=0.46200746, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 52 48 0 100 

python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 6 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 6 --sl 10 --compress-w

# Policy(gpu_batch_size=6, num_gpu_batches=6, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 100 0 100 0
# result
