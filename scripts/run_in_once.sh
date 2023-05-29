model_size=176b
device_names=("NVIDIA_A100-SXM4-80GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_bloom_176b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --time_mult_times 2 \
 --theta 0.0001 --global_bz 32 --debug --group 1 --adapp_group_size 1 --ilp_time_limit 1000 --ilp_tolerance 0.01

