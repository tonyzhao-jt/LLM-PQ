# generate solution
# no nvme for the device
# 0.8 OOM
python3 cost_model.py --model facebook/opt-13b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 32 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 
nvme version
sudo apt-get update & upgrade install nvme-cli
nvme list
# Policy(gpu_batch_size=32, num_gpu_batches=1, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.22422222, cache_cpu_percent=0.77577778, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 22 78 100 0
# script
export TRANSFORMERS_CACHE='***REMOVED***llms/'
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path _DUMMY_ --prompt-len 128 --gen-len 160 \
  --pin-weight 0 --percent 100 0 22 78 100 0 --gpu-batch-size 32 --num-gpu-batches 1 

# run 



# int8
python3 cost_model.py --model facebook/opt-13b --prompt-len 128 --gen-len 160 \
                     --gpu-batch-size 32 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 --compress-w # same to int8
# cannot load dummy, 
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path  _DUMMY_ --prompt-len 128 --gen-len 160 \
  --pin-weight 0 --percent 100 0 100 0 100 0 --gpu-batch-size 32 --num-gpu-batches 1 --compress-w

# model size: 23.921 GB   cache size: 7.031 GB    hidden size (p): 0.088 GB
# peak gpu mem: 14.814 GB projected: False
# prefill latency: 1.711 s        prefill throughput: 2394.367 token/s
# decode latency: 62.488 s        decode throughput: 81.424 token/s
# total latency: 64.198 s total throughput: 79.753 token/s

# 200
python3 cost_model.py --model facebook/opt-13b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 32 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 
# Policy(gpu_batch_size=32, num_gpu_batches=1, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.19687805, cache_cpu_percent=0.80312195, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path _DUMMY_ --prompt-len 128 --gen-len 200 \
  --pin-weight 0 --percent 100 0 20 80 100 0 --gpu-batch-size 32 --num-gpu-batches 1

TorchDevice: cuda:0
  cur_mem: 24.4288 GB,  peak_mem: 26.5644 GB
TorchDevice: cpu
  cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
model size: 23.921 GB   cache size: 8.008 GB    hidden size (p): 0.100 GB
peak gpu mem: 26.564 GB projected: False
prefill latency: 1.344 s        prefill throughput: 3047.989 token/s
decode latency: 117.893 s       decode throughput: 54.015 token/s
total latency: 119.237 s        total throughput: 53.675 token/s

python3 cost_model.py --model facebook/opt-13b --prompt-len 128 --gen-len 200 \
                     --gpu-batch-size 32 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 --compress-w # same to int8

python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path  _DUMMY_ --prompt-len 128 --gen-len 200 \
  --pin-weight 0 --percent 100 0 100 0 100 0 --gpu-batch-size 32 --num-gpu-batches 1 --compress-w
  
prefill latency: 1.639 s        prefill throughput: 2499.021 token/s
decode latency: 78.708 s        decode throughput: 80.907 token/s
total latency: 80.347 s total throughput: 79.654 token/s