# generate solution
# use uniform phase micro batch partition
# opt30 layer: 48
NUM_GPU_BATCHES = 4 
GPU_BATCH_SIZE=8 # 32/4
PARTITION_NUMS=4 # 4 gpus
# no nvme for the device
# for P100
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 12 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7
# 0.8 causes OOM in our case.
#Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=0.44571429, w_cpu_percent=0.55428571, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
44 56 0 100 0 100

# for V100
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4

# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 100 0 100 0

# result
rank #3: TorchDevice: cuda:0
rank #3:   cur_mem: 14.4632 GB,  peak_mem: 21.2364 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 55.803 GB  cache size: 24.281 GB   hidden size (prefill): 0.253 GB
peak gpu mem: 21.236 GB
prefill latency: 97.75 s        prefill throughput: 167.61 token/s
decode latency: 735.81 s        decode throughput: 3.44 token/s
total latency: 833.56 s total throughput: 3.07 token/s


python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 12 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7 --compress-w


python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --compress-w

# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.65405405, cache_cpu_percent=0.34594595, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 65 35 0 100
rank #3: TorchDevice: cuda:0
rank #3:   cur_mem: 4.0755 GB,  peak_mem: 11.3396 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 55.803 GB  cache size: 24.281 GB   hidden size (prefill): 0.253 GB
peak gpu mem: 11.340 GB
prefill latency: 89.49 s        prefill throughput: 183.08 token/s
decode latency: 281.02 s        decode throughput: 9.00 token/s
total latency: 370.52 s total throughput: 6.91 token/s


# 100
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 12 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7

# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=0.44571429, w_cpu_percent=0.55428571, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 45 55 0 100 0 100


python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 12 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7 --compress-w
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.63267974, cache_cpu_percent=0.36732026, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 63 37 0 100


python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4

# 100 0 100 0 100 0

Initializing distributed environment at net-g1:1234, world_size=4, rank=3, local_rank=0.
rank #3: Finished initializing distributed environment
rank #3: model size: 55.803 GB, cache size: 25.102 GB, hidden size (prefill): 0.261 GB
rank #3: warmup - generate
rank #3: benchmark - generate
rank #3: (32, 612) None
rank #3: TorchDevice: cuda:0
rank #3:   cur_mem: 14.4632 GB,  peak_mem: 21.4415 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 55.803 GB	cache size: 25.102 GB	hidden size (prefill): 0.261 GB
peak gpu mem: 21.441 GB
prefill latency: 99.73 s	prefill throughput: 164.29 token/s
decode latency: 1248.43 s	decode throughput: 2.54 token/s
total latency: 1348.16 s	total throughput: 2.37 token/s
Initializing distributed environment at net-g1:1234, world_size=4, rank=3, local_rank=0.
rank #3: Finished initializing distributed environment
rank #3: model size: 55.803 GB, cache size: 25.102 GB, hidden size (prefill): 0.261 GB
rank #3: warmup - generate
rank #3: benchmark - generate
rank #3: (32, 612) None
rank #3: TorchDevice: cuda:0
rank #3:   cur_mem: 4.0755 GB,  peak_mem: 11.5447 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 55.803 GB	cache size: 25.102 GB	hidden size (prefill): 0.261 GB
peak gpu mem: 11.545 GB
prefill latency: 89.65 s	prefill throughput: 182.75 token/s
decode latency: 358.53 s	decode throughput: 8.84 token/s
total latency: 448.18 s	total throughput: 7.14 token/s