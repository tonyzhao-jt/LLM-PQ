#!/bin/bash
# usage: ./test_plan_acc_cmd.sh -n mymodel -s 125m -f /path/to/folder -g 2
# Set default values
model_storage_path='/data/llms/'
# Check if the LLM_PATH environmental variable is set
if [ -n "$LLM_PATH" ]; then
    model_storage_path="$LLM_PATH"
fi
CURRENT_FOLDER=$PWD
device_info=""
model_name="opt"
model_size="125m"
sol_folder="${ROOT_DIR}/scripts/part_strategy"
folder_abs_path="${ROOT_DIR}/scripts/accuracy/bit_for_gptq_test/"
available_methods=('adaqpipe')
cuda_visible_devices="3"
zeroshot=false
export TRANSFORMERS_CACHE=$model_storage_path
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices

mixed_mode=false
adafile_mode=false
wbit=8
storage_path='./tmp'
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
        --mixed)
        mixed_mode=true
        shift
        ;;
        --wbit)
        wbit="$2"
        shift
        shift
        ;;
        --adafile)
        adafile_mode=true
        shift
        ;;
        --storage_path)
        storage_path="$2"
        shift
        shift
        ;;
        -p|--folder_abs_path)
        folder_abs_path="$2"
        shift
        shift
        ;;
        -m|--available_methods)
        available_methods_str="$2"
        IFS=',' read -ra available_methods <<< "$available_methods_str"
        shift
        shift
        ;;
        --zeroshot)
        zeroshot=true
        shift
        ;;
        *)    # unknown option
        shift # past argument
        ;;
    esac
done


# Create storage path directory if it does not exist
if [ ! -d "$storage_path" ]; then
    mkdir -p "$storage_path"
fi

# Run commands that use the input arguments and default values
echo "Model name: $model_name"
echo "Model size: $model_size"
echo "Solution folder: $sol_folder"
echo "CUDA visible devices: $cuda_visible_devices"
echo "Mixed mode: $mixed_mode"
echo "Run zeroshot?: $zeroshot"
if [ ! -d "$adafile" ]; then 
    echo "use file provided mixed-precision setups under folder ${folder_abs_path}"
else 
    echo "Weight bit: $wbit"
fi 
echo "Storage path: $storage_path"

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

# create the corresponding adabit files
if [ "$adafile_mode" = true ]; then
    if [ -n "$device_info" ]; then
        python3 convert_sol_to_gptq_bits.py --model-name ${model_name} --model-size ${model_size} \
                --device-info ${device_info} --sol-folder ${sol_folder}
    fi
fi
cd ${ROOT_DIR}/3rd_party/gptq # main folder
for i in "${!available_methods[@]}"
do  
    if [ "$mixed_mode" = true ]; then
        echo "Mixed mode is enabled"
        file_name="${storage_path}/${model_name}_${model_size}_${device_info}_mixed_${wbit}.pkl"
        # if [ -f $file_abs_path ]; then
        python3 ${model_name}.py ${pretrained_config} c4 --wbits ${wbit} --mixed-bit | tee "${file_name}.txt"
    elif [ "$adafile_mode" = true ]; then
        echo "Adabits file loaded"
        echo "run ${available_methods[i]} perplexity accuracy test"
        if [ -n "$device_info" ]; then
            file_name="${available_methods[i]}_${model_name}_${model_size}_${device_info}_acc_test.pkl"
        else 
            if [ -n "$TEST_RATIO" ]; then 
                file_name="${available_methods[i]}_${model_name}_${model_size}_${TEST_RATIO}_bit_ass.pkl"
            else
                file_name="${available_methods[i]}_${model_name}_${model_size}_bit_ass.pkl"
            fi 
        fi 
        file_abs_path="${folder_abs_path}${file_name}"
        if [ -e "$file_abs_path" ]; then
            echo "File exists!"
            echo $file_abs_path
        else
            echo "File does not exist."
            echo $file_abs_path
            exit 1
        fi
        if [ "$zeroshot" = true ]; then
            # zeroshot test
            cd zeroShot
            python3 main.py ${pretrained_config} c4 --wbits 4 --task piqa,arc_easy,lambada \
             --ada-file ${file_abs_path} 2>&1 | tee "${storage_path}/${file_name}.txt"
        else
            # pp text
            python3 ${model_name}.py ${pretrained_config} c4 --wbits ${wbit} \
            --ada-file ${file_abs_path} 2>&1 | tee "${storage_path}/${file_name}.txt"
        fi
    else
        # if [ -f $file_abs_path ]; then
        file_name="${storage_path}/${model_name}_${model_size}_${device_info}_${wbit}.pkl"
        python3 ${model_name}.py ${pretrained_config} c4 --wbits ${wbit} | tee "${file_name}.txt"
    fi
done
cd $CURRENT_FOLDER # back to current folder