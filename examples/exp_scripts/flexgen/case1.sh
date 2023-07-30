# generate solution
# no nvme for the device
# 0.8 OOM
python3 cost_model.py --model facebook/opt-13b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 32 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 
nvme version
sudo apt-get update & upgrade install nvme-cli
nvme list
# Policy(gpu_batch_size=32, num_gpu_batches=1, w_gpu_percent=0.97080702, w_cpu_percent=0.029192982, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 97 3 0 100 100 0
# script
export TRANSFORMERS_CACHE='/mnt/bn/zjtnaslq/llms/'
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path _DUMMY_ --prompt-len 512 --gen-len 80 \
  --pin-weight 0 --percent 97 3 0 100 100 0 --gpu-batch-size 32 --num-gpu-batches 1 

# run 
element_size = tensor.storage().element_size()
TorchDevice: cuda:0
  cur_mem: 24.4288 GB,  peak_mem: 27.0931 GB
TorchDevice: cpu
  cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
model size: 23.921 GB   cache size: 14.453 GB   hidden size (p): 0.181 GB
peak gpu mem: 27.093 GB projected: False
prefill latency: 8.094 s        prefill throughput: 2024.131 token/s
decode latency: 136.035 s       decode throughput: 18.583 token/s
total latency: 144.130 s        total throughput: 17.762 token/s


# int8
python3 cost_model.py --model facebook/opt-13b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 32 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 --compress-w # same to int8
# cannot load dummy, 
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path /mnt/bn/zjtnaslq/llms --prompt-len 512 --gen-len 80 \
  --pin-weight 0 --percent 100 0 100 0 100 0 --gpu-batch-size 32 --num-gpu-batches 1 --compress-w

# 100
python3 cost_model.py --model facebook/opt-13b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 32 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 
nvme version
sudo apt-get update & upgrade install nvme-cli
nvme list
# Policy(gpu_batch_size=32, num_gpu_batches=1, w_gpu_percent=0.97080702, w_cpu_percent=0.029192982, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 97 3 0 100 100 0
# script
export TRANSFORMERS_CACHE='/mnt/bn/zjtnaslq/llms/'
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path _DUMMY_ --prompt-len 512 --gen-len 100 \
  --pin-weight 0 --percent 97 3 0 100 100 0 --gpu-batch-size 32 --num-gpu-batches 1 

# run 
TorchDevice: cuda:0
  cur_mem: 24.4288 GB,  peak_mem: 27.0931 GB
TorchDevice: cpu
  cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
model size: 23.921 GB   cache size: 14.941 GB   hidden size (p): 0.187 GB
peak gpu mem: 27.093 GB projected: False
prefill latency: 8.060 s        prefill throughput: 2032.750 token/s
decode latency: 166.818 s       decode throughput: 18.991 token/s
total latency: 174.878 s        total throughput: 18.298 token/s
# int8
python3 cost_model.py --model facebook/opt-13b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 32 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 --compress-w # same to int8
# cannot load dummy, 
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path /mnt/bn/zjtnaslq/llms --prompt-len 512 --gen-len 100 \
  --pin-weight 0 --percent 100 0 100 0 100 0 --gpu-batch-size 32 --num-gpu-batches 1 --compress-w

TorchDevice: cuda:0
  cur_mem: 6.8823 GB,  peak_mem: 24.0339 GB
TorchDevice: cpu
  cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
model size: 23.921 GB   cache size: 14.941 GB   hidden size (p): 0.187 GB
peak gpu mem: 24.034 GB projected: False
prefill latency: 8.317 s        prefill throughput: 1969.933 token/s
decode latency: 41.886 s        decode throughput: 75.634 token/s
total latency: 50.203 s total throughput: 63.741 token/s