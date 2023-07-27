model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(4 4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --gamma 1 \
 --fname-suffix "noq_1024_32" \
 --theta 1 --global_bz 32 --debug --fit --s 1024 --n 32


model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(4 4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --gamma 1 \
 --fname-suffix "noq_512_32" \
 --theta 1 --global_bz 32 --debug --fit --s 512 --n 32

model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(4 4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --gamma 1 \
 --fname-suffix "noq_128_32" \
 --theta 1 --global_bz 32 --debug --fit --s 128 --n 32

