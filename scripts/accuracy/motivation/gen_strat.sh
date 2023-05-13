# model_name='bloom'
# model_size="1b1"
model_name='opt'
model_size="1.3b"
ind_abs_path="/workspace/qpipe/scripts/accuracy/generated_ind/"
ind_file_name="${ind_abs_path}gen_${model_name}_${model_size}_ind.pkl"
ind_types=("constant" "uniform" "mag")
for i in "${!ind_types[@]}"
do
    python3 rand_quant.py --model_name $model_name --model_size $model_size -it ${ind_types[i]} --ratio 0.3 \
        --file_name $ind_file_name
done