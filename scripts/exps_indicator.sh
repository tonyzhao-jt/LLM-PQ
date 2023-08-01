# case 5
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_hess_ind.pkl


shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --ilp_seed 120 \
 --fname-suffix "_random_scheme" \
 --theta 9000 --global_bz 32 --ilp_time_limit 300 --fit --debug
#  282487.61ms

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $HESS_FILE --ilp_seed 120 \
 --fname-suffix "_hess_scheme" \
 --theta 0.08 --global_bz 32 --ilp_time_limit 300 --fit --debug
#  283763

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "_shaq_scheme" \
 --theta 50000 --global_bz 32 --ilp_time_limit 300 --fit --debug

# 284998.36ms

# case 9
# random scheme
model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_hess_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --ilp_seed 120 \
 --fname-suffix "_random_scheme" \
 --theta 2400 --global_bz 32 --ilp_time_limit 300 --fit --debug 
#  410162.80ms

# HESS
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $HESS_FILE --ilp_seed 120 \
 --fname-suffix "_hess_scheme" \
 --theta 0.1 --global_bz 32 --ilp_time_limit 300 --fit --debug
# 418279

# SHAQ
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "_shaq_scheme" \
 --theta 20000 --global_bz 32 --ilp_time_limit 300 --fit --debug 
# 418279



