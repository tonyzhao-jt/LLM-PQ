model_size=13b
device_names=("Tesla_P100-PCIE-12GB") 
device_numbers=(2)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl
# baseline comparisons
python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file  $OMEGA_FILE --fit
