# case 3
# model_size=30b
# device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
# device_numbers=(3 1)  # define device numbers as a list of integers
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

# case 4
model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(2 1)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 0.01 --fit --global_bz 32 --debug --group 4 --ilp_time_limit 60

# notice theta need to be changed
# for 13b
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.001 --fit

# for 30b
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.05 --group 2

# for 66b
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.0001 --group_size 2

# all device support tc
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE \
#  --adabits_tc
