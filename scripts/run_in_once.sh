
# model_size=66b
# device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
# device_numbers=(2 2)  # define device numbers as a list of integers
# OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
# HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_hess_ind.pkl
# GROUP_SIZE=1
# ABLATION_FOLDER="${PWD}/ablation/"
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
#  --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"


model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_hess_ind.pkl
GROUP_SIZE=1
ABLATION_FOLDER="${PWD}/ablation/"
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_${GROUP_SIZE}_random" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# ABLATION_FOLDER="${PWD}/ablation/"
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.01 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
#  --fname-suffix "group_${GROUP_SIZE}_hess" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"
# # 215051.94ms


# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
#  --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"
