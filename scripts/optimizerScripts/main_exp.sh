# check whether gurobi is under the path /opt/gurobi/gurobi.lic
# if not alert
if [ ! -f /opt/gurobi/gurobi.lic ]; then
    echo "Gurobi license file not found under /opt/gurobi/gurobi.lic"
    echo "Please make sure that the gurobi license file is under /opt/gurobi/gurobi.lic"
    exit 1
fi

###############################
#
# Main experiment case 1-10
#
###############################
# case 1
model_size=13b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_13b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --fit 


# case 2
model_size=13b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_13b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "lat" \
 --theta 1 --fit --global_bz 32 --fit --s 128 --n 200


# case 3
model_size=30b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_30b_ind.pkl

ABLATION_FOLDER="${PWD}/ablation/"
# mkdir if not exist
mkdir -p $ABLATION_FOLDER

GROUP_SIZE=1
llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_time_limit 60 \
 --theta 50000  --global_bz 32 --group $GROUP_SIZE --debug --fit \
 --fname-suffix "group${GROUP_SIZE}" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"


# case 4
model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_30b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --llm_pq-efficient \
 --theta 1000  --global_bz 32 --debug --ilp_time_limit 160 --fit

# case 5
# we recommend to use Hess for cluster 5, you can get better result than in paper.
model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_ind.pkl
HESS_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_hess_ind.pkl

# llm_pq-algo --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --llm_pq-efficient \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 50 --global_bz 32 --ilp_time_limit 160 --fit --debug 

# HESS file result is much better in this case
llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --llm_pq-efficient \
 --omega_file $HESS_FILE --ilp_seed 120 \
 --theta 50 --global_bz 32 --ilp_time_limit 160 --fit --debug 

# case6
model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_ind.pkl
GROUP_SIZE=1
ABLATION_FOLDER="${PWD}/ablation/"
llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 100 --global_bz 32 --ilp_time_limit 60 --group $GROUP_SIZE --fit --debug \
 --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_size}_GROUP_SIZE${GROUP_SIZE}"

# case 7
model_size=176b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(4 4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_bloom_176b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --time_mult_times 2 \
 --theta 10 --global_bz 32 --ilp_time_limit 1200 --debug  --fit 

# case 8
model_size=176b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-80GB") 
device_numbers=(4 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_bloom_176b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --time_mult_times 2 \
 --theta 10 --global_bz 32 --ilp_time_limit 300 --debug --fit 

# case9
model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_30b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --ilp_time_limit 160 --group 1 --fit --debug  

# case10
model_size=66b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_ind.pkl
# use llm_pq-efficient for this case
llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --llm_pq-efficient \ 
 --theta 1 --global_bz 32 --ilp_time_limit 160 --group 1  --fit 

# case11
model_size=176b
device_names=("NVIDIA_A100-SXM4-80GB") 
device_numbers=(4)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_bloom_176b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --model_name bloom \
 --llm_pq-efficient \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --time_mult_times 2 \
 --theta 10 --global_bz 32 --debug --group 1 --ilp_time_limit 100


##########################
# Latency Aware Tasks
# case 1, 4, 6
##########################

model_size=13b
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_13b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --fname-suffix "lat" \
 --theta 1 --fit --global_bz 32 --use_profiler_prediction --s 128 --n 200

model_size=30b
device_names=("Tesla_P100-PCIE-12GB" "Tesla_V100-SXM2-32GB") 
device_numbers=(3 1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_30b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --llm_pq-efficient \
 --fname-suffix "lat" \
 --theta 1000 --global_bz 32 --debug --ilp_time_limit 160 --s 128 --n 200

model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 150 \
 --llm_pq-efficient \
 --fname-suffix "lat" \
 --theta 100 --global_bz 32 --ilp_time_limit 200 --debug  --s 128 --n 200 


model_size=66b
device_names=("Tesla_V100-SXM2-32GB" "NVIDIA_A100-SXM4-40GB") 
device_numbers=(2 2)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_66b_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 150 \
 --llm_pq-efficient \
 --fname-suffix "lat_eff" \
 --theta 10 --global_bz 32 --ilp_time_limit 200 --debug  --s 128 --n 200 
