export CUDA_VISIBLE_DEVICES=2
available_methods=('adabits')
STRAT_FILE_NAME="sols_opt_13b_NVIDIA_A100-SXM4-40GB_1"
mkdir SINGLE_NODE_PERF_RESULT
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=1 --nproc_per_node=1 --master_port 6666 main.py \
    --model_name opt --model_size 13b --method ${available_methods[i]} \
    --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./SINGLE_NODE_PERF_RESULT/${available_methods[i]}_13b.txt"
    pkill torchrun
    pkill python3
done

