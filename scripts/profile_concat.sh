# for model_size in 13b 30b 66b 
# do
#     python3 profile_concat.py --step 20 --warmup 2 --repeat 10 --model-size $model_size
# done
export CUDA_VISIBLE_DEVICE=1
for model_size in 30b 66b 
do
    python3 profile_concat.py --prompt_length 1024 --step 4 --warmup 2 --repeat 10 --model-size $model_size
done


# for model_size in 176b
# do
#     python3 profile_concat.py --step 20 --warmup 2 --repeat 10 --model-name bloom --model-size $model_size
# done




