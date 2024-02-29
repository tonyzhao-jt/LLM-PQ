export CUDA_VISIBLE_DEVICES=2,3
export TOKENIZERS_PARALLELISM=0
export OMP_NUM_THREADS=32
available_methods=('adabits')
STRAT_FILE_NAME="sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2"
model_size="66b"
# exp1
# rank=1
# MASTER_ADDR=***REMOVED***
# model_size="66b"
# for i in "${!available_methods[@]}"
# do  
#     torchrun --nnodes=2 --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port 6666 --node_rank=${rank} main_p2p.py \
#      --model_name opt --model_size ${model_size} --method ${available_methods[i]} --strat_file_name $STRAT_FILE_NAME \
#         2>&1 | tee "${available_methods[i]}_${model_size}"
#     pkill torchrun
#     pkill python3
# done

rank=0
MASTER_ADDR=net-g15
available_methods=('adaqpipe')
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=2 --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port 6666 --node_rank=${rank} main_p2p.py \
     --model_name opt --model_size ${model_size} --method ${available_methods[i]} --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "${available_methods[i]}_${model_size}"
    pkill torchrun
    pkill python3
done
