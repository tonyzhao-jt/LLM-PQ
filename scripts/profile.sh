# prompts much larger than the generated tokens
export CUDA_VISIBLE_DEVICES=0 # use last one

# python3 profile_lat.py --batch-size 16 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b
# python3 profile_lat.py --batch-size 8 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b
# python3 profile_lat.py --batch-size 4 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b
# python3 profile_lat.py --batch-size 2 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b
# python3 profile_lat.py --batch-size 1 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-name bloom --model-size 176b


# prefill
for batch_size in 1 2 4 8 16 32
do
    for prompt_length in 512 128 
    do
        for model_size in 13b 30b 66b
        do
            python3 profile_lat.py --batch-size $batch_size --input-seq-length $prompt_length --past-seq-length 0 \
             --generated-seq-length 1 --step 20 --warmup 2 --repeat 10 --model-size $model_size
        done
    done
done


# decode
for batch_size in 32 16 8 4 2 1
do
    for past_seq_length in 128 512
    do
        for model_size in 13b 30b 66b
        do
            python3 profile_lat.py --batch-size $batch_size --past-seq-length $past_seq_length \
             --generated-seq-length 100 --step 20 --warmup 2 --repeat 10 --model-size $model_size
        done
    done
done

echo "run 176b"
# for model 176b
for batch_size in 32 16 8 4 2 1
do
    for past_seq_length in 128 512
    do
        # decode
        python3 profile_lat.py --batch-size $batch_size --past-seq-length $past_seq_length \
            --generated-seq-length 100 --step 20 --warmup 2 --repeat 10 --model-name bloom --model-size 176b
        # prefill
        python3 profile_lat.py --batch-size $batch_size --input-seq-length $past_seq_length --past-seq-length 0 \
            --generated-seq-length 1 --step 20 --warmup 2 --repeat 10 --model-name bloom --model-size 176b
    done
done
