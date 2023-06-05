export CUDA_VISIBLE_DEVICES=3
for batch_size in 5 9 14
do
    for past_seq_length in 512
    do
        for model_size in 30b
        do
            python3 profile_lat.py --batch-size $batch_size --past-seq-length $past_seq_length \
             --generated-seq-length 100 --step 30 --warmup 2 --repeat 10 --model-size $model_size
        done
    done
done
