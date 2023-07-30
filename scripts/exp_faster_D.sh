
# Grouping
# model_size=30b
# device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
# device_numbers=(3 1)  # define device numbers as a list of integers
# OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
# GROUP_SIZE=4

# ABLATION_FOLDER="${PWD}/ablation/"
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_time_limit 300 \
#  --theta 1  --global_bz 32 --group $GROUP_SIZE --debug --fit \
#  --fname-suffix "group${GROUP_SIZE}" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"
# # Shaq time:  0.5376548767089844
# # latency: 101407.4030524407

GROUP_SIZE=2
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_time_limit 300 \
 --theta 1  --global_bz 32 --group $GROUP_SIZE --debug --fit \
 --fname-suffix "group${GROUP_SIZE}" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# shaq time 1.1731576919555664
# lat: 49322.43213030226

GROUP_SIZE=1
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_time_limit 60 \
 --theta 1  --global_bz 32 --group $GROUP_SIZE --debug --fit \
 --fname-suffix "group${GROUP_SIZE}" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"
# shaq time 63.34078502655029
# lat 49322.43213030226

# SHAQH
ABLATION_FOLDER="${PWD}/ablation/"
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_time_limit 60\
 --shaq-efficient \
 --theta 1  --global_bz 32 --group $GROUP_SIZE --debug --fit \
 --fname-suffix "group_shaqh" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# Shaq time:  5.784833192825317
# lat 53588.10561712076


# cluster 6
# model_size=66b
# device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
# device_numbers=(2 2)  # define device numbers as a list of integers
# OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
# GROUP_SIZE=4
# ABLATION_FOLDER="${PWD}/ablation/"
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
#  --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# # shaq result: Minimax Lat 40061.88301815567
# # Shaq time:  0.8628251552581787

model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
GROUP_SIZE=2
ABLATION_FOLDER="${PWD}/ablation/"
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# shaq result: Minimax Lat 42658.35716035665
# Shaq time:  2.6208925247192383

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
# shaq result: Minimax Lat 42412.9236576391
# 19.142317056655884


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
 --shaq-efficient \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_shaqh" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# shaq result: Minimax Lat 68064.056671993
# 7.702705383300781


# model_size=30b
# device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
# device_numbers=(3 1)  # define device numbers as a list of integers
# OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
# GROUP_SIZE=4
# ABLATION_FOLDER="${PWD}/ablation/"
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 --group $GROUP_SIZE \
#  --theta 1  --global_bz 32 --debug --ilp_time_limit 160 --fit \
#  --fname-suffix "group_shaqh" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# # shaq result: Minimax Lat 108140.82173228034
# # Shaq time:  0.6258668899536133


model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
GROUP_SIZE=2
ABLATION_FOLDER="${PWD}/ablation/"
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 --group $GROUP_SIZE \
 --theta 1  --global_bz 32 --debug --ilp_time_limit 160 --fit \
 --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"
# shaq result: Minimax Lat 127310.98415288287
# Shaq time:  8.147095203399658

model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
GROUP_SIZE=1
ABLATION_FOLDER="${PWD}/ablation/"
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 --group $GROUP_SIZE \
 --theta 1  --global_bz 32 --debug --ilp_time_limit 60 --fit \
 --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"
# shaq result: Minimax Lat 133693.0892058463
# Shaq time:  126.26175141334534


model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
GROUP_SIZE=1
ABLATION_FOLDER="${PWD}/ablation/"
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --shaq-efficient \
 --theta 1  --global_bz 32 --debug --ilp_time_limit 60 --fit \
 --fname-suffix "group_shaq" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"
# shaq result: Minimax Lat 108140.82173228034
# Shaq time:  2.005858898162842


# GROUP_SIZE=4
# ABLATION_FOLDER="${PWD}/ablation/"
# # uniform cases
# # case10
# model_size=66b
# device_names=("Tesla_V100-SXM2-32GB") 
# device_numbers=(4)  # define device numbers as a list of integers
# OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
# # use shaq-efficient for this case
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --omega_file $OMEGA_FILE --ilp_seed 120 --group $GROUP_SIZE \
#  --theta 1  --global_bz 32 --debug --ilp_time_limit 60 --fit \
#  --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# shaq result: Minimax Lat 137367.9335484014
# Shaq time:  0.8970189094543457

model_size=66b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
# use shaq-efficient for this case
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --omega_file $OMEGA_FILE --ilp_seed 120 --group $GROUP_SIZE \
 --theta 1  --global_bz 32 --debug --ilp_time_limit 60 --fit \
 --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# shaq result: Minimax Lat 127606.2088761086
# Shaq time:  5.81243896484375


GROUP_SIZE=1
ABLATION_FOLDER="${PWD}/ablation/"
# uniform cases
# case10
model_size=66b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
# use shaq-efficient for this case
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --omega_file $OMEGA_FILE --ilp_seed 120 --group $GROUP_SIZE \
 --theta 1  --global_bz 32 --debug --ilp_time_limit 60 --fit \
 --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"
# shaq result: Minimax Lat 112720.37227891953
# Shaq time:  104.73534512519836


# GROUP_SIZE=1
# ABLATION_FOLDER="${PWD}/ablation/"
# # uniform cases
# # case10
# model_size=66b
# device_names=("Tesla_V100-SXM2-32GB") 
# device_numbers=(4)  # define device numbers as a list of integers
# OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
# # use shaq-efficient for this case
# shaq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 --group $GROUP_SIZE \
#  --shaq-efficient \
#  --theta 1  --global_bz 32 --debug --ilp_time_limit 60 --fit \
#  --fname-suffix "group_shaqh" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

shaq result: Minimax Lat 110881.82803846375
Shaq time:  2.110968828201294




# D 
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --shaq-efficient \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 10000 --global_bz 32 --group 1 --ilp_time_limit 160 --fit --debug 


model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl

shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --shaq-efficient \
 --force-fixed-D \
 --fname-suffix "_forced_D" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 10000 --global_bz 32 --group 1 --ilp_time_limit 160 --fit --debug 



model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
GROUP_SIZE=1
ABLATION_FOLDER="${PWD}/ablation/"
shaq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --force-fixed-D \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --shaq-efficient \
 --theta 1  --global_bz 32 --debug --ilp_time_limit 60 --fit \
 --fname-suffix "force-fixed-D" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"