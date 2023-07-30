# generate solution
# no nvme for the device
python3 cost_model.py --model facebook/opt-13b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 32 --gpu-mem 40 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1
nvme version
sudo apt-get update & upgrade install nvme-cli
nvme list
# Policy(gpu_batch_size=32, num_gpu_batches=1, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.39783784, cache_cpu_percent=0.60216216, act_gpu_percent=1.0, act_cpu_percent=0.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 40 60 100 0
export TRANSFORMERS_CACHE='/data/llms/'
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path _DUMMY_ --prompt-len 512 --gen-len 80 --gpu-batch-size 32 --num-gpu-batches 1 \
  --pin-weight 0 --percent 100 0 40 60 100 0 


warnings.warn(
TorchDevice: cuda:0
  cur_mem: 24.4288 GB,  peak_mem: 32.5612 GB
TorchDevice: cpu
  cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
model size: 23.921 GB   cache size: 14.453 GB   hidden size (p): 0.181 GB
peak gpu mem: 32.561 GB projected: False
prefill latency: 2.920 s        prefill throughput: 5610.932 token/s
decode latency: 52.450 s        decode throughput: 48.199 token/s
total latency: 55.370 s total throughput: 46.235 token/s

python3 cost_model.py --model facebook/opt-13b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 32 --gpu-mem 40 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 1 --compress-w

python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path /data/llms --prompt-len 512 --gen-len 80 \
  --pin-weight 0 --percent 100 0 100 0 100 0 --gpu-batch-size 32 --num-gpu-batches 1 --compress-w


# 100
python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path _DUMMY_ --prompt-len 512 --gen-len 100 --gpu-batch-size 32 --num-gpu-batches 1 \
  --pin-weight 0 --percent 100 0 40 60 100 0 

TorchDevice: cuda:0
  cur_mem: 24.4288 GB,  peak_mem: 32.7178 GB
TorchDevice: cpu
  cur_mem: 0.0000 GB,  peak_mem: 0.0000 GB
model size: 23.921 GB   cache size: 14.941 GB   hidden size (p): 0.187 GB
peak gpu mem: 32.718 GB projected: False
prefill latency: 3.074 s        prefill throughput: 5330.035 token/s
decode latency: 68.018 s        decode throughput: 46.576 token/s
total latency: 71.092 s total throughput: 45.012 token/s

python3 -m flexgen.flex_opt --model facebook/opt-13b \
 --path _DUMMY_ --prompt-len 512 --gen-len 100 \
  --pin-weight 0 --percent 100 0 100 0 100 0 --gpu-batch-size 32 --num-gpu-batches 1 --compress-w
