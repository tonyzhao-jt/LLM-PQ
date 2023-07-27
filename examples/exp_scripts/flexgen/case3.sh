# generate solution
# use uniform phase micro batch partition
# opt30 layer: 48
NUM_GPU_BATCHES = 4 
GPU_BATCH_SIZE=8 # 32/4
PARTITION_NUMS=4 # 4 gpus
# no nvme for the device
# for T4
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 50 \
                     --gpu-batch-size 8 --gpu-mem 16 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 --partition_nums 4
Policy(gpu_batch_size=4, num_gpu_batches=8, w_gpu_percent=0.87170068, w_cpu_percent=0.12829932, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))

# for V100
python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 50 \
                     --gpu-batch-size 4 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 8 --partition_nums 4

# Policy(gpu_batch_size=32, num_gpu_batches=1, w_gpu_percent=0.48772552, w_cpu_percent=0.51227448, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# script
export TRANSFORMERS_CACHE='/data/llms/'
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path _DUMMY_ --prompt-len 512 --gen-len 100 \
  --pin-weight 0 --percent 49 51 0 100 0 100 --gpu-batch-size $GPU_BATCH_SIZE --num-gpu-batches $NUM_GPU_BATCHES --debug fewer_batch 


TorchDevice: cuda:0
  cur_mem: 11.9449 GB,  peak_mem: 15.0242 GB
TorchDevice: cpu
  cur_mem: 12.6802 GB,  peak_mem: 0.0000 GB
model size: 23.921 GB   cache size: 14.941 GB   hidden size (p): 0.187 GB
peak gpu mem: 15.024 GB projected: True
prefill latency: 7.099 s        prefill throughput: 2307.857 token/s
decode latency: 161.668 s       decode throughput: 19.596 token/s
total latency: 168.767 s        total throughput: 18.961 token/s