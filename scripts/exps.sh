# case 1
model_size=13b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 0.001 --fit --global_bz 32 --use_profiler_prediction

# case 2
model_size=13b
device_names=("NVIDIA_A100-SXM4-40GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 0.001 --fit --global_bz 32 --ilp_time_limit 30

# model_size=13b
# device_names=("Tesla_P100-PCIE-12GB") 
# device_numbers=(2)  # define device numbers as a list of integers
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.001 --fit

# case 2-backup
# model_size=13b
# device_names=("Tesla_T4") 
# device_numbers=(2)  # define device numbers as a list of integers
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl

# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.001 --fit

# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.001

# case 3
model_size=30b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 0.1 --fit --global_bz 32 --debug --group 4 --adapp_group_size 2 --ilp_time_limit 1000 


# case 4
model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --fit --global_bz 32 --debug --group 1 --adapp_group_size 1 --ilp_time_limit 200 --ilp_tolerance 0.015 

# backup 4-1
# model_size=30b
# device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
# device_numbers=(1 1)  # define device numbers as a list of integers
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

# case 5
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --group 2 --ilp_time_limit 160 --fit --debug --ilp_tolerance 0.04

# case6
model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --group 8 --ilp_time_limit 200 --fit --debug --ilp_tolerance 0.07

# case 7
model_size=176b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(4 4)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_bloom_176b_ind.pkl

python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 0.01 --global_bz 32 --group 2 --adapp_group_size 2 --ilp_time_limit 60  --ilp_tolerance 0.01

# case 8
model_size=176b
device_names=("Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-80GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_bloom_176b_ind.pkl

# uniform cases
# case9
model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

# case9 backup
# model_size=30b
# device_names=("Tesla_T4") 
# device_numbers=(3)  # define device numbers as a list of integers
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

# uniform cases
# case10
model_size=66b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

# case11
model_size=176b
device_names=("Tesla_V100-SXM2-80GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_176b_ind.pkl


# check case
model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(2 1)  # define device numbers as a list of integers
OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

# python3 valid.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file  $OMEGA_FILE --debug

# check device order
python3 check_order.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --debug \
 --test_method 'adaqpipe'
