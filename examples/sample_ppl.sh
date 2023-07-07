sample_acc_dir="${PWD}/sample_acc"
mkdir $sample_acc_dir

cd "$ROOT_DIR/scripts/accuracy"
model_name='opt'
model_size="13b"
folder_abs_path="${ROOT_DIR}/scripts/accuracy/bit_for_gptq_test"

device_info="NVIDIA_A100-SXM4-40GB_1"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    -p $folder_abs_path --device_info $device_info --adafile -m "uniform","adaqpipe", "pipeedge"