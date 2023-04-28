# model_name='bloom'
# model_size="1b1"
model_name='opt'
model_size="1.3b"
ind_types=("constant" "uniform")
for i in "${!ind_types[@]}"
do
python rand_quant.py --model_name $model_name --model_size $model_size -it ${ind_types[i]}
done