available_methods=('adabits' 'adaqpipe' 'pipeedge' 'uniform')
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=1 --nproc_per_node=2 --master_port 6666 main_p2p.py --model_name opt --model_size 13b --method ${available_methods[i]} \
        2>&1 | tee "${available_methods[i]}_13b"
    pkill torchrun
    pkill python3
done


