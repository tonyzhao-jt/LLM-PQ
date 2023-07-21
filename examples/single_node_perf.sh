method='adabits' # from adabits, shaq, uniform, pipeedge
STRAT_FILE_NAME="sols_opt_13b_Tesla_V100-SXM2-32GB_1"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE='***REMOVED***llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
# CPU Not Enough
export LOAD_IN_NP="1"
MODEL_NAME='opt'
MODEL_SIZE="13b"
mkdir SINGLE_NODE_PERF_RESULT

shaq-dist --master_port 6666 \
    --model_name $MODEL_NAME --model_size $MODEL_SIZE\
    --method $method \
    --strat_file_name $STRAT_FILE_NAME \
    2>&1 | tee "./SINGLE_NODE_PERF_RESULT/${available_methods[i]}_13b.txt"
pkill torchrun
pkill python3

# Other Notice: 
    # Please run weight_convert.py first
    # Please run fake_calib (refer to readme)

# checker
# python3 check_strat.py --model_name opt --model_size 13b --method ${available_methods[i]} \
#     --strat_file_name $STRAT_FILE_NAME 


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
