model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
root_path="${ROOT_DIR}"
model_size="30b"
# model_name="bloom"
# model_prefix="bigscience/bloom-"
# pretrained_config="bigscience/bloom"
model_name="opt"
model_prefix="facebook/opt-"
pretrained_config="facebook/opt-30b"
device_info="Tesla_T4_4_rand"
available_methods=('adaqpipe')
folder_abs_path="${root_path}/scripts/accuracy/bit_for_gptq_test/"
export CUDA_VISIBLE_DEVICES=1
# create the corresponding files
python3 convert_sol_to_gptq_bits.py --model-name ${model_name} --model-size ${model_size} \
        --device-info ${device_info}
cd ${root_path}/3rd_party/gptq/zeroShot
for i in "${!available_methods[@]}"
do  
    echo "run ${available_methods[i]} accuracy test"
    file_name="${available_methods[i]}_${model_size}_${device_info}_acc_test.pkl"
    file_abs_path="${folder_abs_path}${file_name}"
    if [ -f $file_abs_path ]; then
        python3 main.py ${pretrained_config} c4 --wbits 4 --task piqa,arc_easy,lambada \
        --ada-file ${file_abs_path} 2>&1 | tee "${file_name}.txt"
    else
        echo "$file_abs_path does not exist"
    fi

done