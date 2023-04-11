# prompts smaller to generated tokens
python3 profile_layer_device_result.py --past-seq-length 64 --generated-seq-length 100 --step 20 --shard 0 
python3 profile_layer_device_result.py --past-seq-length 64 --generated-seq-length 100 --step 20 --shard 1
# prompts compatable to generated tokens
python3 profile_layer_device_result.py --past-seq-length 128 --generated-seq-length 100 --step 20 --shard 0 
python3 profile_layer_device_result.py --past-seq-length 128 --generated-seq-length 100 --step 20 --shard 1
# prompts much larger than the generated tokens
python3 profile_layer_device_result.py --past-seq-length 512 --generated-seq-length 100 --step 20 --shard 0 
python3 profile_layer_device_result.py --past-seq-length 512 --generated-seq-length 100 --step 20 --shard 1
# test 2048 also (larger pastseq)
python3 profile_layer_device_result.py --past-seq-length 2048 --generated-seq-length 100 --step 20 --shard 0
python3 profile_layer_device_result.py --past-seq-length 2048 --generated-seq-length 100 --step 20 --shard 1