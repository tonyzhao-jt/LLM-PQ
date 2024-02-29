# generate solution
# use uniform phase micro batch partition
# opt30 layer: 48
NUM_GPU_BATCHES = 4 
GPU_BATCH_SIZE=8 # 32/4
PARTITION_NUMS=4 # 4 gpus
# no nvme for the device
# for T4
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 15 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7
# 0.8 causes OOM in our case.
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=0.62857143, w_cpu_percent=0.37142857, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 62 38 0 100 0 100

# for V100
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4

# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 100 0 100 0


rank #3: Finished initializing distributed environment
rank #3: model size: 55.803 GB, cache size: 24.281 GB, hidden size (prefill): 0.253 GB
rank #3: warmup - generate
rank #3: benchmark - generate
rank #3: (32, 592) None
/usr/local/lib/python3.9/dist-packages/torch/distributed/distributed_c10d.py:293: UserWarning: torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead
  warnings.warn(
rank #3: TorchDevice: cuda:0
rank #3:   cur_mem: 14.4632 GB,  peak_mem: 21.2364 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 55.803 GB  cache size: 24.281 GB   hidden size (prefill): 0.253 GB
peak gpu mem: 21.236 GB
prefill latency: 23.08 s        prefill throughput: 709.83 token/s
decode latency: 742.88 s        decode throughput: 3.40 token/s
total latency: 765.96 s total throughput: 3.34 token/s


python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 15 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7 --compress-w


python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --compress-w

# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=-7.9193765e-15, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 100 0 0 100
rank #3: TorchDevice: cuda:0
rank #3:   cur_mem: 4.0755 GB,  peak_mem: 11.3396 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 55.803 GB  cache size: 24.281 GB   hidden size (prefill): 0.253 GB
peak gpu mem: 11.340 GB
prefill latency: 18.89 s        prefill throughput: 867.24 token/s
decode latency: 225.32 s        decode throughput: 11.22 token/s
total latency: 244.21 s total throughput: 10.48 token/s




# 100
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 15 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7
# 63 37 0 100 0 100
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4
# 100 0 100 0 100 0


python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 15 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7 --compress-w

# 100 0 97 3 0 100
# 100 0 100 0 100 0