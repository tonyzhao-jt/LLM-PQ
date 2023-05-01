cd /workspace/qpipe/3rd_party/gptq/zeroShot
# model_prefix="facebook/opt-"
# model_name='opt'
# model_size="1.3b"
model_prefix="bigscience/bloom-"
model_name='bloom'
model_size="1b1"
available_methods=("constant" "uniform")
available_methods=("mag")
folder_abs_path="/workspace/qpipe/scripts/accuracy/motivation/bit_result/"
CUDA_VISIBLE_DEVICES=0
for i in "${!available_methods[@]}"
do  
    file_name="${available_methods[i]}_${model_name}_${model_size}_bit_ass.pkl"
    file_abs_path="${folder_abs_path}${file_name}"
    echo "run ${available_methods[i]}"
    echo "python3 main.py ${model_prefix}${model_size} c4 --wbits 4 --task piqa --ada-file ${file_abs_path} 2>&1 | tee "${file_name}.txt""
    python3 main.py ${model_prefix}${model_size} c4 --wbits 4 --task piqa --ada-file ${file_abs_path} 2>&1 | tee "${file_name}.txt"
done