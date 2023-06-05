model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 0.001 --global_bz 32 --group 1 --ilp_time_limit 200 --debug  --s 128 --n 200 --ilp_tolerance 0.03
