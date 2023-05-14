# prompts much larger than the generated tokens
export CUDA_VISIBLE_DEVICES=3 # use last one

# python3 profile_lat.py --batch-size 16 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b
# python3 profile_lat.py --batch-size 8 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b
# python3 profile_lat.py --batch-size 4 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b
# python3 profile_lat.py --batch-size 2 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b
# python3 profile_lat.py --batch-size 1 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b

for batch_size in 16 8 4 2 1
do
    for past_seq_length in 128 512
    do
        for model_size in 13b 30b 66b
        do
            python3 profile_lat.py --batch-size $batch_size --past-seq-length $past_seq_length \
             --generated-seq-length 500 --step 100 --repeat 20 --model-size $model_size
        done
    done
done