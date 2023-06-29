model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
model_name="opt"
model_prefix="facebook/opt-"
model_size="13b"
device_info="Tesla_V100-SXM2-32GB_4"
available_methods=('adabits' 'adaqpipe' 'pipeedge' 'uniform')
folder_abs_path="/workspace/qpipe/scripts/accuracy/bit_for_gptq_test/"
export CUDA_VISIBLE_DEVICES=0
# create the corresponding files
# python3 convert_sol_to_gptq_bits.py --model-name ${model_name} --model-size ${model_size} \
#         --device-info ${device_info}
cd /workspace/qpipe/3rd_party/gptq/zeroShot
python3 main.py ${model_prefix}${model_size} c4 --wbits 4 --task piqa,arc_easy,lambada \
--sq-test 2>&1 | tee "sq_test.txt"
# for i in "${!available_methods[@]}"
# do  
#     echo "run ${available_methods[i]} accuracy test"
#     file_name="${available_methods[i]}_${model_size}_${device_info}_acc_test.pkl"
#     file_abs_path="${folder_abs_path}${file_name}"
#     if [ -f $file_abs_path ]; then

#     else
#         echo "$file_abs_path does not exist"
#     fi

# done