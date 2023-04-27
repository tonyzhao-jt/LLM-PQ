

model_sizes=("30b" "66b" "175b" "560m")
model_names=("opt" "opt" "opt" "bloom")
for i in "${!model_sizes[@]}"
do  
    python3 test_gen_fake_calib.py --model-name "${model_names[i]}" --model-size "${model_sizes[i]}"
done