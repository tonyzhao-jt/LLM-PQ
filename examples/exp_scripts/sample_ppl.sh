# required
model_name='opt'
model_size="13b"
user_abs_file_path="sols_opt_13b_Tesla_V100-SXM2-32GB_1"
# setups
export CUDA_VISIBLE_DEVICES=2
sample_acc_dir="${PWD}/sample_acc"
mkdir $sample_acc_dir
cd "$ROOT_DIR/scripts/accuracy"

# dont change this part
device_info="Tesla_V100-SXM2-32GB_1"
folder_abs_path="${ROOT_DIR}/scripts/accuracy/bit_for_gptq_test"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    -p $folder_abs_path --adafile -m "uniform","shaq","pipeedge","adabits" \
     --user_abs_file_path $user_abs_file_path --device_info $device_info

