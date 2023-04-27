cd /workspace/qpipe/3rd_party/gptq/zeroShot
model_prefix="facebook/opt-"
model_size="30b"
device_info="Tesla_T4_2_Tesla_V100-SXM2-32GB_1"
available_methods=("pipeedge" "qpipe" "uniform_2" "uniform_4" "adabit")
folder_abs_path="/workspace/qpipe/scripts/accuracy/bit_for_gptq_test/"
CUDA_VISIBLE_DEVICES=0
for i in "${!available_methods[@]}"
do  
    file_name="${available_methods[i]}_${model_size}_${device_info}_acc_test.pkl"
    file_abs_path="${folder_abs_path}${file_name}"
    echo "run ${available_methods[i]}"
    python3 main.py ${model_prefix}${model_size} c4 --wbits 4 --task piqa \
        --ada-file ${file_abs_path} 2>&1 | tee "${file_name}.txt"
done