# generate solution
# use uniform phase micro batch partition
NUM_GPU_BATCHES = 4 
GPU_BATCH_SIZE=8 # 32/4
PARTITION_NUMS=4 # 4 gpus
# no nvme for the device
# for A100
python3 cost_model.py --model facebook/opt-66b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 8 --gpu-mem 40 --cpu-mem 212 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 
# 0.8 causes OOM in our case.
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.26890432, cache_cpu_percent=0.73109568, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 27 73 0 100

# for V100
python3 cost_model.py --model facebook/opt-66b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=0.81042034, w_cpu_percent=0.18957966, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 81 19 0 100 0 100

# result
rank #3: TorchDevice: cuda:1
rank #3:   cur_mem: 31.2498 GB,  peak_mem: 32.7306 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 122.375 GB cache size: 20.250 GB   hidden size (prefill): 0.158 GB
peak gpu mem: 32.731 GB
prefill latency: 5.45 s prefill throughput: 751.80 token/s
decode latency: 314.95 s        decode throughput: 16.16 token/s
total latency: 320.40 s total throughput: 15.98 token/s



python3 cost_model.py --model facebook/opt-66b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 8 --gpu-mem 40 --cpu-mem 212 --nvme-mem 0 \
                      --num-gpu-batches 4  --partition_nums 4 --compress-w

# 100 0 100 0 100 0

python3 cost_model.py --model facebook/opt-30b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4  --partition_nums 4 --compress-w

# 100 0 100 0 100 0
rank #3: TorchDevice: cuda:1
rank #3:   cur_mem: 8.7987 GB,  peak_mem: 15.3223 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 122.375 GB cache size: 20.250 GB   hidden size (prefill): 0.158 GB
peak gpu mem: 15.322 GB
prefill latency: 5.42 s prefill throughput: 755.81 token/s
decode latency: 296.45 s        decode throughput: 17.16 token/s
total latency: 301.87 s total throughput: 16.96 token/s


# 200
python3 cost_model.py --model facebook/opt-66b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 8 --gpu-mem 40 --cpu-mem 212 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 
# 100 0 24 76 0 100

python3 cost_model.py --model facebook/opt-66b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 8 --gpu-mem 40 --cpu-mem 212 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --compress-w

# 100 0 100 0 100 0


python3 cost_model.py --model facebook/opt-66b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4

# 81 19 0 100 0 100

python3 cost_model.py --model facebook/opt-66b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --compress-w
# 100 0 100 0 100 0