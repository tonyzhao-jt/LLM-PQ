
export TRANSFORMERS_CACHE='/data/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
# CPU Not Enough
export LOAD_IN_NP="1"

# prefill
for batch_size in 1 2 4 8 16 32
do
    for prompt_length in 512 256 128 
    do
        for model_size in 13b 30b 66b
        do
            python3 profile_lat.py --batch-size $batch_size --input-seq-length $prompt_length --past-seq-length 0 \
             --generated-seq-length 1 --step 20 --warmup 2 --repeat 5 --model-size $model_size --num-stacks 10
        done
    done
done


# decode
# for batch_size in 1 2 4 8 16 32
# do
#     for past_seq_length in 128 256 512
#     do
#         for model_size in 13b 30b 66b
#         do
#             python3 profile_lat.py --batch-size $batch_size --past-seq-length $past_seq_length \
#              --generated-seq-length 100 --step 20 --warmup 2 --repeat 20 --model-size $model_size
#         done
#     done
# done

# echo "run 176b"
# # for model 176b
# for batch_size in 1 2 4 8 16 32
# do
#     for past_seq_length in 128 256 512
#     do
#         # decode
#         python3 profile_lat.py --batch-size $batch_size --past-seq-length $past_seq_length \
#             --generated-seq-length 100 --step 20 --warmup 2 --repeat 10 --model-name bloom --model-size 176b
#         # prefill
#         python3 profile_lat.py --batch-size $batch_size --input-seq-length $past_seq_length --past-seq-length 0 \
#             --generated-seq-length 1 --step 20 --warmup 2 --repeat 10 --model-name bloom --model-size 176b
#     done
# done