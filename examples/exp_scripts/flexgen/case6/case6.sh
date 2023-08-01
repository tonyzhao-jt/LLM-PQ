# generate solution
# use uniform phase micro batch partition
NUM_GPU_BATCHES = 4 
GPU_BATCH_SIZE=8 # 32/4
PARTITION_NUMS=4 # 4 gpus
# no nvme for the device
# for A100
python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 40 --cpu-mem 212 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 
# 0.8 causes OOM in our case.
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=1.0, w_cpu_percent=0.0, cache_gpu_percent=0.034534535, cache_cpu_percent=0.96546547, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 100 0 3 97 0 100

# for V100
python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4
# Policy(gpu_batch_size=8, num_gpu_batches=4, w_gpu_percent=0.77272193, w_cpu_percent=0.22727807, cache_gpu_percent=0.0, cache_cpu_percent=1.0, act_gpu_percent=0.0, act_cpu_percent=1.0, overlap=True, sep_layer=False, pin_weight=False, cpu_cache_compute=True, attn_sparsity=1, compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False, enabled=True), compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False, enabled=True))
# 77 23 0 100 0 100

# result



python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 40 --cpu-mem 212 --nvme-mem 0 \
                      --num-gpu-batches 4  --partition_nums 4 --compress-w

# 100 0 100 0 100 0

python3 cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 80 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4  --partition_nums 4 --compress-w

# 100 0 100 0 100 0

# 100


python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 40 --cpu-mem 212 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 

# 100 0 3 97 0 100

python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 40 --cpu-mem 212 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --compress-w
# 100 0 100 0 100 0

python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4

# 77 23 0 100 0 100

python3 cost_model.py --model facebook/opt-66b --prompt-len 512 --gen-len 100 \
                     --gpu-batch-size 8 --gpu-mem 32 --cpu-mem 200 --nvme-mem 0 \
                      --num-gpu-batches 4 --partition_nums 4 --compress-w

# 100 0 100 0 100 0