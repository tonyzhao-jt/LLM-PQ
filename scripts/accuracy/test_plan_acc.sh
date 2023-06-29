model_storage_path='/data/llms/'
# Check if the LLM_PATH environmental variable is set
if [ -n "$LLM_PATH" ]; then
    model_storage_path="$LLM_PATH"
fi

export TRANSFORMERS_CACHE=$model_storage_path
export CUDA_VISIBLE_DEVICES=3 # control the gpu used
device_info="Tesla_T4_4_rand"
available_methods=('adaqpipe')
# model control
model_name="opt"
model_size="1.3b"
model_name="bloom"
model_size="1b1"

sol_folder="${ROOT_DIR}/scripts/part_strategy" # control the part_strategy folder path

if [[ "${model_name}" == "opt" ]]; then
    model_prefix="facebook/opt-"
    pretrained_config="facebook/opt-${model_size}"
else
    if [[ "${model_size}" != "176b" ]]; then
    pretrained_config="bigscience/bloom-${model_size}"
    else
    pretrained_config="bigscience/bloom"
    fi
fi

folder_abs_path="${ROOT_DIR}/scripts/accuracy/bit_for_gptq_test/"
# create the corresponding files
python3 convert_sol_to_gptq_bits.py --model-name ${model_name} --model-size ${model_size} \
        --device-info ${device_info} --sol-folder ${sol_folder}
cd ${ROOT_DIR}/3rd_party/gptq # main folder
for i in "${!available_methods[@]}"
do  
    echo "run ${available_methods[i]} perplexity accuracy test"
    file_name="${available_methods[i]}_${model_size}_${device_info}_acc_test.pkl"
    file_abs_path="${folder_abs_path}${file_name}"
    # if [ -f $file_abs_path ]; then
    python3 ${model_name}.py ${pretrained_config} c4 --wbits 8 \
    --ada-file ${file_abs_path} 2>&1 | tee "${file_name}.txt"
    # else
    #     echo "$file_abs_path does not exist"
    # fi
done