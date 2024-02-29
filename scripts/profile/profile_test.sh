# prompts much larger than the generated tokens
export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
# CPU Not Enough
export LOAD_IN_NP="1"
cd $ROOT_DIR/scripts/profile
export CUDA_VISIBLE_DEVICES=1 # use last one

# prefill
for batch_size in 3
do
    for prompt_length in 384
    do
        for model_size in 125m
        do
            python3 profile_lat.py --batch-size $batch_size --input-seq-length $prompt_length --past-seq-length 0 \
             --generated-seq-length 1 --step 20 --warmup 2 --repeat 10 --model-size $model_size
        done
    done
done

