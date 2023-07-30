# VISIBLE DEVIES
export CUDA_VISIBLE_DEVICES=3
CUR_DIR=${PWD}

# required
model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
sol_folder="/data/llms/sols"
# setups
sample_acc_dir="${PWD}/sample_acc"
mkdir $sample_acc_dir
cd "$ROOT_DIR/scripts/accuracy"
cd rand && bash update.sh 
cd ..

# test config
model_name='opt'
model_size="13b" # must match with the following, else error.

bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    --model_storage_path $model_storage_path --wbit 16

# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
#     --model_storage_path $model_storage_path --wbit 8

model_size="30b" # must match with the following, else error.
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    --model_storage_path $model_storage_path --wbit 16

# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
#     --model_storage_path $model_storage_path --wbit 8

model_size="66b" # must match with the following, else error.
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    --model_storage_path $model_storage_path --wbit 16

bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    --model_storage_path $model_storage_path --wbit 8