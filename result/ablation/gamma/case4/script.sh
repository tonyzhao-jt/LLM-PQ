export CUDA_VISIBLE_DEVICES=2,3
export TOKENIZERS_PARALLELISM=0
export OMP_NUM_THREADS=32
STRAT_FILE_NAME="sols_opt_30b_Tesla_P100-PCIE-12GB_3_Tesla_V100-SXM2-32GB_1"
available_methods=('adabits')
rank=2
model_size="30b"
MASTER_ADDR=net-g1
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=3 --nproc_per_node=1 --master_addr ${MASTER_ADDR} --master_port 6666 --node_rank=${rank} main_p2p.py \
     --model_name opt --model_size ${model_size} --method ${available_methods[i]} --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "${available_methods[i]}_${model_size}"
    pkill torchrun
    pkill python3
done

# export CUDA_VISIBLE_DEVICES=2,3
# export TOKENIZERS_PARALLELISM=0
# export OMP_NUM_THREADS=32
# available_methods=('adaqpipe')
# rank=0
# model_size="30b"
# MASTER_ADDR=***REMOVED***
# STRAT_FILE_NAME="sols_opt_30b_Tesla_P100-PCIE-12GB_3_Tesla_V100-SXM2-32GB_1"
# for i in "${!available_methods[@]}"
# do  
#     torchrun --nnodes=3 --nproc_per_node=1 --master_addr ${MASTER_ADDR} --master_port 6666 --node_rank=${rank} main_p2p.py \
#      --model_name opt --model_size ${model_size} --method ${available_methods[i]} --strat_file_name $STRAT_FILE_NAME \
#         2>&1 | tee "${available_methods[i]}_${model_size}"
#     pkill torchrun
#     pkill python3
# done