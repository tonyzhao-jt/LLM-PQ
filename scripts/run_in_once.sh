model_size=13b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_13b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --fit 