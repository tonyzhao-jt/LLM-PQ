# prompts much larger than the generated tokens
cd $ROOT_DIR/scripts
export CUDA_VISIBLE_DEVICES=0 # use last one

# prefill
for batch_size in 3 5 7
do
    for prompt_length in 768 384
    do
        for model_size in 13b 30b 66b
        do
            python3 profile_lat.py --batch-size $batch_size --input-seq-length $prompt_length --past-seq-length 0 \
             --generated-seq-length 1 --step 20 --warmup 2 --repeat 10 --model-size $model_size
        done
    done
done

# decode
for batch_size in 3 5 7
do
    for past_seq_length in 768 384
    do
        for model_size in 13b 30b 66b
        do
            python3 profile_lat.py --batch-size $batch_size --past-seq-length $past_seq_length \
             --generated-seq-length 100 --step 20 --warmup 2 --repeat 10 --model-size $model_size
        done
    done
done
