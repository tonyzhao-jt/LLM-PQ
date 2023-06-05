model_sizes=("350m" "560m" "1.3b" "13b" "30b" "66b" "176b" )
model_names=("opt" "bloom"  "opt" "opt" "opt" "opt" "bloom")
for i in "${!model_sizes[@]}"
do  
    python3 test_gen_fake_calib.py --model-name "${model_names[i]}" --model-size "${model_sizes[i]}"
done