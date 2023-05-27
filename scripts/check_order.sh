model_size=176b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(4 4)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_bloom_176b_ind.pkl

python3 check_order.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --ilp_seed 120 --model_name bloom \
 --theta 0.01 --global_bz 32 --debug --ilp_time_limit 60 --test_method 'adaqpipe'