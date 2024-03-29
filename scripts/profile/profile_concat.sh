export CUDA_VISIBLE_DEVICES=0

for model_size in 13b 30b 66b; do
    python3 profile_concat.py --step 20 --warmup 2 --repeat 10 --model-size $model_size
done

for model_size in 176b; do
    python3 profile_concat.py --step 20 --warmup 2 --repeat 10 --model-name bloom --model-size $model_size
done




