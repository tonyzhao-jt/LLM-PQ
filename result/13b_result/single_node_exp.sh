export CUDA_VISIBLE_DEVICES=3
available_methods=('adabits')
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=1 --nproc_per_node=1 --master_port 6666 main_p2p.py --model_name opt --model_size 13b --method ${available_methods[i]} \
        2>&1 | tee "${available_methods[i]}_13b"
    pkill torchrun
    pkill python3
done


