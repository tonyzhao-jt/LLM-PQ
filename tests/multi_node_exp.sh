available_methods=('adabits' 'adaqpipe' 'pipeedge' 'uniform')
rank=0
model_size="13b"
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=2 --nproc_per_node=1 --master_addr 10.139.117.21 --master_port 6666 --node_rank=${rank} main_p2p.py \
     --model_name opt --model_size ${model_size} --method ${available_methods[i]} \
        2>&1 | tee "${available_methods[i]}_${model_size}"
    pkill torchrun
    pkill python3
done

rank=1
available_methods=('adaqpipe')
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=2 --nproc_per_node=1 --master_addr 10.128.101.150 --master_port 6666 --node_rank=${rank} main_p2p.py \
     --model_name opt --model_size ${model_size} --method ${available_methods[i]} \
        2>&1 | tee "${available_methods[i]}_${model_size}"
    pkill torchrun
    pkill python3
done


