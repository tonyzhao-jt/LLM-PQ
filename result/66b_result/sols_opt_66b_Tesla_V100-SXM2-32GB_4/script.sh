# export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=0
export OMP_NUM_THREADS=32
available_methods=('uniform')
for i in "${!available_methods[@]}"
do  
    torchrun --nnodes=1 --nproc_per_node=4 --master_port 6666 main_p2p.py --model_name opt --model_size 66b --method ${available_methods[i]} \
        2>&1 | tee "${available_methods[i]}_66b_homo"
    pkill torchrun
    pkill python3
done


