
# required
model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
model_name='opt'
model_size="66b" # must match with the following, else error.
user_abs_file_path="sols_opt_66b_Tesla_V100-SXM2-32GB_2_NVIDIA_A100-SXM4-40GB_2"
# setups
export CUDA_VISIBLE_DEVICES=3
sample_acc_dir="${PWD}/sample_acc"
mkdir $sample_acc_dir
cd "$ROOT_DIR/scripts/accuracy"
cd rand && bash update.sh 
cd ..

# dont change this part
device_info="Tesla_V100-SXM2-32GB_1"
folder_abs_path="${ROOT_DIR}/scripts/accuracy/bit_for_gptq_test"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    -p $folder_abs_path --adafile -m "llm_pq","adabits" \
     --user_abs_file_path $user_abs_file_path --device_info $device_info --model_storage_path $model_storage_path




# # required
# model_storage_path='/data/llms/'
# export TRANSFORMERS_CACHE=$model_storage_path
# model_name='opt'
# model_size="30b" # must match with the following, else error.
# user_abs_file_path="sols_opt_30b_Tesla_P100-PCIE-12GB_3_Tesla_V100-SXM2-32GB_1"
# # setups
# export CUDA_VISIBLE_DEVICES=2
# sample_acc_dir="${PWD}/sample_acc"
# mkdir $sample_acc_dir
# cd "$ROOT_DIR/scripts/accuracy"
# cd rand && bash update.sh 
# cd ..

# # dont change this part
# device_info="Tesla_V100-SXM2-32GB_1"
# folder_abs_path="${ROOT_DIR}/scripts/accuracy/bit_for_gptq_test"
# bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
#     -p $folder_abs_path --adafile -m "uniform","llm_pq","pipeedge","adabits" \
#      --user_abs_file_path $user_abs_file_path --device_info $device_info --model_storage_path $model_storage_path

