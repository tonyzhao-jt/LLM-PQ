#!/bin/bash
# usage: ./test_plan_acc_cmd.sh -n mymodel -s 125m -f /path/to/folder -g 2
# Set default values
model_storage_path='/data/llms/'
device_info="Tesla_T4_4_rand"
available_methods=('adaqpipe')
model_name="opt"
model_size="125m"
sol_folder="${ROOT_DIR}/scripts/part_strategy"
cuda_visible_devices="3"
export TRANSFORMERS_CACHE=$model_storage_path
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -n|--model_name)
        model_name="$2"
        shift
        shift
        ;;
        -s|--model_size)
        model_size="$2"
        shift
        shift
        ;;
        -f|--sol_folder)
        sol_folder="$2"
        shift
        shift
        ;;
        -g|--cuda_visible_devices)
        cuda_visible_devices="$2"
        shift
        shift
        ;;
        *)    # unknown option
        shift # past argument
        ;;
    esac
done

# Run commands that use the input arguments and default values
echo "model_name is set to: $model_name"
echo "model_size is set to: $model_size"
echo "sol_folder is set to: $sol_folder"
echo "cuda_visible_devices is set to: $cuda_visible_devices"

export TRANSFORMERS_CACHE=$model_storage_path
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices

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