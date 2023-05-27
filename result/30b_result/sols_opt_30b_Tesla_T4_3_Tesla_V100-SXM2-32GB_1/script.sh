export TOKENIZERS_PARALLELISM=0
export OMP_NUM_THREADS=32
model_size="30b"
STRAT_FILE_NAME="sols_opt_30b_Tesla_T4_3_Tesla_V100-SXM2-32GB_1"
# exp1
rank=1
MASTER_ADDR=<YOURADDR>
available_methods=('adabits')
# for i in "${!available_methods[@]}"
# do  
#     torchrun --nnodes=2 --nproc_per_node=1 --master_addr $MASTER_ADDR --master_port 6666 --node_rank=${rank} main_p2p.py \
#      --model_name opt --model_size ${model_size} --method ${available_methods[i]} --strat_file_name $STRAT_FILE_NAME \
#         2>&1 | tee "${available_methods[i]}_${model_size}"
#     pkill torchrun
#     pkill python3
# done

# exp2
rank=0
MASTER_ADDR=<YOURADDR>
available_methods=('adaqpipe')
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=2 --nproc_per_node=1 --master_addr $MASTER_ADDR --master_port 6666 --node_rank=${rank} main_p2p.py \
     --model_name opt --model_size ${model_size} --method ${available_methods[i]} --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "${available_methods[i]}_${model_size}"
    pkill torchrun
    pkill python3
done

