model_size=13b
device_names=("NVIDIA_A100-SXM4-40GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl

python3 check_order.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 0.01 --fit --global_bz 32 --debug --group 4 --ilp_time_limit 60 --test_method 'adaqpipe'