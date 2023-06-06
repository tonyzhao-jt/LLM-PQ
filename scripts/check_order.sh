model_size=30b
device_names=("Tesla_T4") 
device_numbers=(4) # define device numbers as a list of integers
# python3 check_order.py --model_size ${model_size} \
#  --device_names "${device_names[@]}" \
#  --device_numbers "${device_numbers[@]}" \
#  --ilp_seed 120 \
#  --theta 0.001 --fit --global_bz 32 --use_profiler_prediction --s 128 --n 200


python3 check_order.py --fname "/workspace/qpipe/result/ind_result/30b/sols_opt_30b_Tesla_T4_4_rand.pkl" --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --ilp_seed 120 \
 --theta 0.001 --fit --global_bz 32 --use_profiler_prediction --s 128 --n 200