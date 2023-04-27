#!/bin/bash
# Set the list of model sizes
model_sizes=("30b" "30b" \
            "66b" "66b" \
            "175b" "175b")

# Set the list of device names and numbers for each model size
device_names=("Tesla_T4 Tesla_V100-SXM2-32GB" "Tesla_T4 Tesla_V100-SXM2-32GB" \
              "Tesla_T4 Tesla_V100-SXM2-32GB" "Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB" \
              "Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB" "Tesla_V100-SXM2-32GB A100-SXM-80GB")
device_numbers=("2 1" "4 1" \
                "4 2" "4 2" \
                "4 4" "8 4")

# Loop over each model size and launch the script with the corresponding device names and numbers
for i in "${!model_sizes[@]}"
do  

    # Split the device numbers into a list of integers
    IFS=' ' read -r -a device_numbers_list <<< "${device_numbers[i]}"
    # Split the device names into a list of strings
    IFS=' ' read -r -a device_names_list <<< "${device_names[i]}"

    # make folder to store the comm model
    python comm_model_helper.py --model_size "${model_sizes[i]}" \
                      --device_names "${device_names_list[@]}" \
                      --device_numbers "${device_numbers_list[@]}"

    # Launch the Python script with the corresponding arguments
    python uniform.py --model_size "${model_sizes[i]}" \
                      --device_names "${device_names_list[@]}" \
                      --device_numbers "${device_numbers_list[@]}"
done
