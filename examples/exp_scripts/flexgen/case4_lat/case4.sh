# generate solution
# use uniform phase micro batch partition
# opt30 layer: 48
NUM_GPU_BATCHES = 4 
GPU_BATCH_SIZE=8 # 32/4
PARTITION_NUMS=4 # 4 gpus
# no nvme for the device
# for P100
python3 cost_model.py --model facebook/opt-30b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 8 --gpu-mem 12 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7
# 0.8 causes OOM in our case.
#Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=0.51357143, w_cpu_percent=0.48642857, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 51 49 0 100 0 100

# for V100
python3 cost_model.py --model facebook/opt-30b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4

# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 100 0 100 0

# result
rank #3: TorchDevice: cuda:0
rank #3:   cur_mem: 14.4632 GB,  peak_mem: 17.5737 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 55.803 GB  cache size: 11.812 GB   hidden size (prefill): 0.123 GB
peak gpu mem: 17.574 GB
prefill latency: 28.69 s        prefill throughput: 142.75 token/s
decode latency: 1378.84 s       decode throughput: 3.69 token/s
total latency: 1407.53 s        total throughput: 3.64 token/s


python3 cost_model.py --model facebook/opt-30b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 8 --gpu-mem 12 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7 --compress-w


python3 cost_model.py --model facebook/opt-30b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --compress-w
# 100 0 100 0 100 0
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=1.0, cache_cpu_percent=0.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
rank #3: TorchDevice: cuda:0
rank #3:   cur_mem: 4.0755 GB,  peak_mem: 7.9439 GB
rank #3: TorchDevice: cpu
rank #3:   cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
rank #3: model size: 55.803 GB  cache size: 11.812 GB   hidden size (prefill): 0.123 GB
peak gpu mem: 7.944 GB
prefill latency: 21.16 s        prefill throughput: 193.58 token/s
decode latency: 524.57 s        decode throughput: 9.70 token/s
total latency: 545.73 s total throughput: 9.38 token/s


# 200
python3 cost_model.py --model facebook/opt-30b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 8 --gpu-mem 12 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7

# 51 49 0 100 0 100


python3 cost_model.py --model facebook/opt-30b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 8 --gpu-mem 12 --cpu-mem 100 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --alpha-g 0.7 --compress-w
# 100 0 100 0 100 0


python3 cost_model.py --model facebook/opt-30b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4

# 100 0 100 0 100 0
