# 30b cases
# Adaqpipe (default)
model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4)  # define device numbers as a list of integers
# 498507
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_ind.pkl
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 1 --global_bz 32 --ilp_time_limit 160 --group 2 --adapp_group_size 2 --fit --debug --ilp_tolerance 0.21
# # 550842.3445710108
# cp ${ROOT_DIR}/scripts/part_strategy/sols_opt_30b_Tesla_T4_4.pkl \
#  ${ROOT_DIR}/result/ind_result/30b/sols_opt_30b_Tesla_T4_4_ada.pkl

# 514729
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_30b_hess_ind.pkl
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 1 --global_bz 32 --ilp_time_limit 160 --group 2 --adapp_group_size 2 --fit --debug --ilp_tolerance 0.21
# # 567064.2353234964
# cp ${ROOT_DIR}/scripts/part_strategy/sols_opt_30b_Tesla_T4_4.pkl \
#  ${ROOT_DIR}/result/ind_result/30b/sols_opt_30b_Tesla_T4_4_hess.pkl

# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --ilp_seed 120 \
#  --theta 1 --global_bz 32 --ilp_time_limit 160 --group 2 --adapp_group_size 2 --fit --debug --ilp_tolerance 0.23
# # 503387.67
# cp ${ROOT_DIR}/scripts/part_strategy/sols_opt_30b_Tesla_T4_4.pkl \
#  ${ROOT_DIR}/result/ind_result/30b/sols_opt_30b_Tesla_T4_4_rand.pkl


model_size=66b
device_names=("Tesla_T4" "Tesla_V100-SXM2-32GB") 
device_numbers=(4 2)  # define device numbers as a list of integers

# 236258.57056295982
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_66b_ind.pkl
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.1 --global_bz 32 --group 2 --ilp_time_limit 160 --fit --debug --ilp_tolerance 0.15

# cp ${ROOT_DIR}/scripts/part_strategy/sols_opt_66b_Tesla_T4_4_Tesla_V100-SXM2-32GB_2.pkl \
#  ${ROOT_DIR}/result/ind_result/66b/sols_opt_66b_Tesla_T4_4_Tesla_V100-SXM2-32GB_2_ada.pkl


# 246214.395
# OMEGA_FILE=/workspace/qpipe/scripts/accuracy/generated_ind/gen_opt_66b_hess_ind.pkl
# python3 algo_entry.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --omega_file $OMEGA_FILE --ilp_seed 120 \
#  --theta 0.1 --global_bz 32 --group 2 --ilp_time_limit 160 --fit --debug --ilp_tolerance 0.085

# cp ${ROOT_DIR}/scripts/part_strategy/sols_opt_66b_Tesla_T4_4_Tesla_V100-SXM2-32GB_2.pkl \
#  ${ROOT_DIR}/result/ind_result/66b/sols_opt_66b_Tesla_T4_4_Tesla_V100-SXM2-32GB_2_hess.pkl

# 236258.57056295982
python3 algo_entry.py --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --ilp_seed 120 \
 --theta 10 --global_bz 32 --group 2 --ilp_time_limit 160 --fit --debug --ilp_tolerance 0.15

cp ${ROOT_DIR}/scripts/part_strategy/sols_opt_66b_Tesla_T4_4_Tesla_V100-SXM2-32GB_2.pkl \
 ${ROOT_DIR}/result/ind_result/66b/sols_opt_66b_Tesla_T4_4_Tesla_V100-SXM2-32GB_2_rand.pkl
