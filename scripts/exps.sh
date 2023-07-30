# case 1
model_size=13b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --fit 


# case 2
model_size=13b
device_names=("NVIDIA_A100-SXM4-40GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --fit 


# case 3
model_size=30b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

ABLATION_FOLDER="${PWD}/ablation/"

GROUP_SIZE=1
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_time_limit 60 \
 --theta 50000  --global_bz 32 --group $GROUP_SIZE --debug --fit \
 --fname-suffix "group${GROUP_SIZE}" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"


# case 4
model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --shaq-efficient \
 --theta 1000  --global_bz 32 --debug --ilp_time_limit 160 --fit

# case 5
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 5000 --global_bz 32 --group 2 --ilp_time_limit 160 --fit --debug 

# case6
model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 200 --fit --debug

# case 7
model_size=176b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(4 4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_bloom_176b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --time_mult_times 2 \
 --theta 1 --global_bz 32 --ilp_time_limit 1200 --debug  --fit 

# case 8
model_size=176b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-80GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_bloom_176b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --time_mult_times 2 \
 --theta 0.01 --global_bz 32 --group 1 --adapp_group_size 1 --ilp_time_limit 300 --debug  --ilp_tolerance 0.08

# uniform cases
# case9
model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --ilp_time_limit 160 --group 1 --fit --debug  

# uniform cases
# case10
model_size=66b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
# use shaq-efficient for this case
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --shaq-efficient \ 
 --theta 1 --global_bz 32 --ilp_time_limit 160 --group 1  --fit 

# case11
model_size=176b
device_names=("NVIDIA_A100-SXM4-80GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_bloom_176b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --time_mult_times 2 \
 --theta 0.0001 --global_bz 32 --debug --group 5 --adapp_group_size 5 --ilp_time_limit 100 --ilp_tolerance 0.01




# Latency Aware Tasks
# case 1, 4, 6
model_size=13b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "lat" \
 --theta 1 --fit --global_bz 32 --use_profiler_prediction --s 128 --n 200

model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "lat" \
 --theta 1 --global_bz 32 --group 1  --debug --ilp_time_limit 160  --s 128 --n 200

model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 150 \
 --fname-suffix "lat" \
 --theta 1 --global_bz 32 --group 1 --ilp_time_limit 200 --debug  --s 128 --n 200 





# Memory Not Scarce / Varing Prompt Length
# 1024 512 128
model_size=30b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 0.001 --global_bz 32 --group 1 --ilp_time_limit 200 --fit --debug




model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --shaq-efficient \
 --ilp_seed 120 \
 --fname-suffix "lat" \
 --theta 1 --global_bz 32 --group 1  --debug --ilp_time_limit 160  --s 128 --n 200
