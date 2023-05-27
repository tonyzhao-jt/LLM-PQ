export TOKENIZERS_PARALLELISM=0
export OMP_NUM_THREADS=32
available_methods=('uniform')
rank=0
model_size="66b"
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=2 --nproc_per_node=4 --master_addr <> --master_port 6666 --node_rank=${rank} main_p2p.py \
     --model_name opt --model_size ${model_size} --method ${available_methods[i]} \
        2>&1 | tee "${available_methods[i]}_${model_size}"
    pkill torchrun
    pkill python3
done

# rank=1
# available_methods=('adaqpipe')
# for i in "${!available_methods[@]}"
# do  
#     torchrun --nnodes=2 --nproc_per_node=3 --master_addr <> --master_port 6666 --node_rank=${rank} main_p2p.py \
#      --model_name opt --model_size ${model_size} --method ${available_methods[i]} \
#         2>&1 | tee "${available_methods[i]}_${model_size}"
#     pkill torchrun
#     pkill python3
# done

