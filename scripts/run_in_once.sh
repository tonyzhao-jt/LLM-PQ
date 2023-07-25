# model_size=66b
# device_names=("Tesla_V100-SXM2-32GB") 
# device_numbers=(4)  # define device numbers as a list of integers
# OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --shaq-efficient \
#  --theta 1 --global_bz 32 --ilp_time_limit 160 --fit --debug 

model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "lat" \
 --theta 1 --global_bz 32 --group 1  --debug --ilp_time_limit 160  --s 128 --n 200 --ilp_tolerance 0.01