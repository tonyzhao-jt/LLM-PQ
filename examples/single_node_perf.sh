export CUDA_VISIBLE_DEVICES=0
available_methods=('adabits') # from adabits, adaqpipe, uniform, pipeedge
# A100:
# STRAT_FILE_NAME="sols_opt_13b_NVIDIA_A100-SXM4-40GB_1"

# V100: 
# be careful when using V100, you need to recompile the bitsandbytes
# refer LPTorch
STRAT_FILE_NAME="sols_opt_13b_Tesla_V100-SXM2-32GB_1"
export TRANSFORMERS_CACHE='/data/llms/'
mkdir SINGLE_NODE_PERF_RESULT


# Other Notice: 
    # Please run weight_convert.py first
    # Please run fake_calib (refer to readme)

# checker
# python3 check_strat.py --model_name opt --model_size 13b --method ${available_methods[i]} \
#     --strat_file_name $STRAT_FILE_NAME 

# A100-Single node
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=1 --nproc_per_node=1 --master_port 6666 main.py \
    --model_name opt --model_size 13b --method ${available_methods[i]} \
    --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./SINGLE_NODE_PERF_RESULT/${available_methods[i]}_13b.txt"
    pkill torchrun
    pkill python3
done

# V100-Single node
# available_methods=('adaqpipe')
# mkdir SINGLE_NODE_PERF_RESULT
# # ValueError: Please run weight_convert.py first
# for i in "${!available_methods[@]}"
# do  
#     torchrun --nnodes=1 --nproc_per_node=1 --master_port 6666 main.py \
#     --model_name opt --model_size 13b --method ${available_methods[i]} \
#     --strat_file_name $STRAT_FILE_NAME \
#         2>&1 | tee "./SINGLE_NODE_PERF_RESULT/${available_methods[i]}_13b.txt"
#     pkill torchrun
#     pkill python3
# done


# for i in "${!available_methods[@]}"
# do  
#     torchrun --nnodes=1 --nproc_per_node=1 --master_port 6666 main.py \
#     --model_name opt --model_size 13b --method ${available_methods[i]} \
#     --strat_file_name $STRAT_FILE_NAME --perf-mode \
#         2>&1 | tee "./SINGLE_NODE_PERF_RESULT/${available_methods[i]}_13b.txt"
#     pkill torchrun
#     pkill python3
# done
