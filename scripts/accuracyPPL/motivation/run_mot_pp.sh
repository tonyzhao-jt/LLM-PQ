folder_abs_path="${PWD}/bit_result/"
storage_path="${PWD}/ratio_result_pp"
cd ..
# exp1
# model_name='opt'
# model_size="1.3b"
# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $storage_path \
#     -p $folder_abs_path --adafile -m "constant","uniform"

# model_name='bloom'
# model_size="1b1"
# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $storage_path \
#     -p $folder_abs_path --adafile -m "constant","uniform"

# exp1
# model_name='opt'
# model_size="1.3b"
# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $storage_path \
#     -p $folder_abs_path --adafile -m 0,1,2

model_name='bloom'
model_size="3b"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $storage_path \
    -p $folder_abs_path --adafile -m 0,1,2