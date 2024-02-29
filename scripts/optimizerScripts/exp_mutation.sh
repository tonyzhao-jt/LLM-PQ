############################################################################################################################
# This experiment discuss the consistency of the strategy in handling different workload
############################################################################################################################
# base
model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
GROUP_SIZE=1
ABLATION_FOLDER="${PWD}/ablation/"
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

echo $ROOT_DIR # make sure you set this
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" --gamma 0.8\
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_gamma_0.8" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" --gamma 0.6\
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_gamma_0.6" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# latency task
model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 150 \
 --shaq-efficient \
 --fname-suffix "lat" \
 --theta 100 --global_bz 32 --ilp_time_limit 200 --debug  --s 128 --n 200 

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 150 --gamma 0.8 \
 --shaq-efficient \
 --fname-suffix "lat_gamma_0.8" \
 --theta 100 --global_bz 32 --ilp_time_limit 200 --debug  --s 128 --n 200 


shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 150 --gamma 0.6 \
 --shaq-efficient \
 --fname-suffix "lat_gamma_0.6" \
 --theta 100 --global_bz 32 --ilp_time_limit 200 --debug  --s 128 --n 200 




model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
GROUP_SIZE=1
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}"\
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --shaq-efficient \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --s 256 \
 --fname-suffix "group_s_256_n_100_gamma_1" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" --gamma 0.8\
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --s 256 \
 --shaq-efficient \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_s_256_n_100_gamma_0.8" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" --gamma 0.6\
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --s 256 \
 --shaq-efficient \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_s_256_n_100_gamma_0.6" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}"\
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --shaq-efficient \
 --s 256 --n 200 \
 --fname-suffix "group_s_256_n_200_gamma_1" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}"\
 --omega_file $OMEGA_FILE --ilp_seed 120 --gamma 0.8 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --s 256 --n 200 \
 --fname-suffix "group_s_256_n_200_gamma_0.8" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}"\
 --omega_file $OMEGA_FILE --ilp_seed 120 --gamma 0.6 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --shaq-efficient \
 --s 256 --n 200 \
 --fname-suffix "group_s_256_n_200_gamma_0.6" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"