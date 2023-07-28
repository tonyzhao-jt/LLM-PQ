model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_hess_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "_shaq_scheme" \
 --theta 50000 --global_bz 32 --ilp_time_limit 300 --fit --debug 