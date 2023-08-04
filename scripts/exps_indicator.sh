# case 5
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_hess_ind.pkl


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
 --omega_file $HESS_FILE --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_${GROUP_SIZE}_hess" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

#  283394

# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --fname-suffix "_shaq_scheme" \
#  --theta 30000 --global_bz 32 --ilp_time_limit 300 --fit --debug

#  284998.36ms

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
 --theta 80000 --global_bz 32 --ilp_time_limit 300 --fit --debug

# # case 9
# perplexity ok group
# # random scheme
# model_size=30b
# device_names=("Tesla_T4") 
# device_numbers=(4)  # define device numbers as a list of integers
# OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
# HESS_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_hess_ind.pkl

# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --ilp_seed 120 \
#  --fname-suffix "_random_scheme" \
#  --theta 2400 --global_bz 32 --ilp_time_limit 300 --fit --debug 
# #  410162.80ms

# # HESS
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $HESS_FILE --ilp_seed 120 \
#  --fname-suffix "_hess_scheme" \
#  --theta 0.1 --global_bz 32 --ilp_time_limit 300 --fit --debug
# # 418279

# # SHAQ
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --fname-suffix "_shaq_scheme" \
#  --theta 20000 --global_bz 32 --ilp_time_limit 300 --fit --debug 
# # 418279



shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --ilp_seed 120 \
 --fname-suffix "_rand_scheme" \
 --time_mult_times 2 \
 --theta 5300 --global_bz 32 --ilp_time_limit 300 --debug --fit 
# 259769.95589218778