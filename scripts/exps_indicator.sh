# case 5
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_hess_ind.pkl

# random scheme
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --ilp_seed 120 \
 --fname-suffix "_random_scheme" \
 --theta 1000 --global_bz 32 --ilp_time_limit 300 --fit --debug 
# 252149.18ms

# hess
# number scale difference is too large for hess and shaq. need to adjust theta
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $HESS_FILE --ilp_seed 120 \
 --fname-suffix "_hess_scheme" \
 --theta 0.1 --global_bz 32 --ilp_time_limit 300 --fit --debug 
# 249499.14084434888

# shaq
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_hess_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "_shaq_scheme" \
 --theta 50000 --global_bz 32 --ilp_time_limit 300 --fit --debug 
# 252109.91ms

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


# 40061.88



