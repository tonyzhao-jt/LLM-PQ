
export TRANSFORMERS_CACHE='/mnt/bn/zjtnaslq/llms/'
export CALIB_ROOT_FOLDER="${ROOT_DIR}/examples/"
export NP_WEIGHT_FOLDER="${TRANSFORMERS_CACHE}/converted_weights_np/"
export OMP_NUM_THREADS=20
# CPU Not Enough
export LOAD_IN_NP="1"

available_methods=('adabits') # from adabits, shaq, uniform, pipeedge
STRAT_FILE_NAME="sols_opt_30b_Tesla_T4_3_Tesla_V100-SXM2-32GB_1group4"

storage_folder="MULTI_NODE_PERF_RESULT"
mkdir $storage_folder
# A100-Single node
MASTER_ADDR=***REMOVED***
MASTER_PORT=1234
MODEL_SIZE=30b
MODEL_NAME=opt
NNODES=2
NUM_DEVICES=3
RANK=0
# for i in "${!available_methods[@]}"
# do  
    # shaq-dist --nnodes=${NNODES} --nproc_per_node=$NUM_DEVICES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=${RANK} \
    #     --model_name ${MODEL_NAME} --model_size $MODEL_SIZE --method ${available_methods[i]} \
    #     --strat_file_name $STRAT_FILE_NAME \
    #         2>&1 | tee "./${storage_folder}/${available_methods[i]}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}.txt"
    # pkill torchrun
    # pkill python3
# done
for i in "${!available_methods[@]}"
do  
torchrun --nnodes=${NNODES} --nproc_per_node=$NUM_DEVICES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank=${RANK} main.py\
    --model_name ${MODEL_NAME} --model_size $MODEL_SIZE --method ${available_methods[i]} \
    --strat_file_name $STRAT_FILE_NAME \
        2>&1 | tee "./${storage_folder}/${available_methods[i]}_${STRAT_FILE_NAME}_${MODEL_NAME}_${MODEL_SIZE}.txt"
    pkill torchrun
    pkill python3
done


