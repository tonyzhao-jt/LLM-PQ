############################################################################################################################
# ablation test for theta
############################################################################################################################
# case 5 different theta
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "theta_66b_100x" \
 --theta 500000 --global_bz 32 --ilp_time_limit 300 --fit --debug

model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "theta_66b_10x" \
 --theta 50000 --global_bz 32 --ilp_time_limit 300 --fit --debug

model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "theta_66b_1x" \
 --theta 5000 --global_bz 32 --ilp_time_limit 300 --fit --debug


# ablation test 2
model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_30b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "theta_30b_100x" \
 --theta 20000 --global_bz 32 --ilp_time_limit 300 --fit --debug

# 10 times
model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_30b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "theta_30b_10x" \
 --theta 2000 --global_bz 32 --ilp_time_limit 300 --fit --debug

# 100 times
model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_30b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "theta_30b_1x" \
 --theta 200 --global_bz 32 --ilp_time_limit 300 --fit --debug


