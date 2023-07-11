export CUDA_VISIBLE_DEVICES=3
sample_acc_dir="${PWD}/sample_acc"
mkdir $sample_acc_dir

export TRANSFORMERS_CACHE='/data/llms/'
cd "$ROOT_DIR/scripts/accuracy"
model_name='bloom'
model_size="176b"
folder_abs_path="${ROOT_DIR}/scripts/accuracy/bit_for_gptq_test"

device_info="NVIDIA_A100-SXM4-80GB_4"
# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
#     -p $folder_abs_path --device_info $device_info --adafile -m "adabits"

bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    -p $folder_abs_path --device_info $device_info --adafile -m "uniform","adaqpipe","pipeedge","adabits"

device_info="Tesla_V100-SXM2-32GB_4_NVIDIA_A100-SXM4-40GB_4"

bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    -p $folder_abs_path --device_info $device_info --adafile -m "uniform","adaqpipe","pipeedge","adabits"

device_info="Tesla_V100-SXM2-32GB_4_NVIDIA_A100-SXM4-80GB_2"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    -p $folder_abs_path --device_info $device_info --adafile -m "uniform","adaqpipe","pipeedge","adabits"
