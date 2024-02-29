# VISIBLE DEVIES
export CUDA_VISIBLE_DEVICES=3
CUR_DIR=${PWD}

# required
model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
# export HF_DATASETS_CACHE="/data/dataset/"
# sol_folder="/data/sols"
# setups
sample_acc_dir="${PWD}/sample_acc"
mkdir $sample_acc_dir
cd "$ROOT_DIR/scripts/accuracy"
cd rand && bash update.sh 
cd ..

# test config
model_name='opt'
model_size="66b" # must match with the following, else error.
user_abs_file_path="sols_opt_66b_Tesla_T4_4_Tesla_V100-SXM2-32GB_2_random_scheme"
# dont change this part
device_info="Tesla_V100-SXM2-32GB_1"
folder_abs_path="${ROOT_DIR}/scripts/accuracy/bit_for_gptq_test"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $sample_acc_dir \
    -p $folder_abs_path --adafile -m "llm_pq" \
     --user_abs_file_path $user_abs_file_path --device_info $device_info --model_storage_path $model_storage_path 
