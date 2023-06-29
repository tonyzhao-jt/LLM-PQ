storage_path="${PWD}/motive_mix"
CUDA_VISIBLE_DEVICES=2
# test
model_name=bloom
model_size="560m"
# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 3 --storage_path $storage_path
# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 3 --mixed --storage_path $storage_path
# run multiple mixed-precision result
export LP_BITS_THRESHOLD=6.0
model_name=bloom
model_size="3b"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 3 --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 3 --mixed --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 4 --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 4 --mixed --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 8 --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 16 --storage_path $storage_path

model_name=opt
model_size="1.3b"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 3 --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 3 --mixed --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 4 --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 4 --mixed --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 8 --storage_path $storage_path
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --wbit 16 --storage_path $storage_path
