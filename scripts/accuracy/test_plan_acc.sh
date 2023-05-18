model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
model_name="opt"
model_prefix="facebook/opt-"
model_size="13b"
device_info="Tesla_P100-PCIE-12GB_2"
available_methods=('adabits' 'adaqpipe' 'pipeedge' 'uniform')
folder_abs_path="/workspace/qpipe/scripts/accuracy/bit_for_gptq_test/"
CUDA_VISIBLE_DEVICES=0
# create the corresponding files
python3 convert_sol_to_gptq_bits.py --model-name ${model_name} --model-size ${model_size} \
        --device-info ${device_info}
cd /workspace/qpipe/3rd_party/gptq/zeroShot
for i in "${!available_methods[@]}"
do  
    echo "run ${available_methods[i]} accuracy test"
    file_name="${available_methods[i]}_${model_size}_${device_info}_acc_test.pkl"
    file_abs_path="${folder_abs_path}${file_name}"
    python3 main.py ${model_prefix}${model_size} c4 --wbits 4 --task piqa \
        --ada-file ${file_abs_path} 2>&1 | tee "${file_name}.txt"
done