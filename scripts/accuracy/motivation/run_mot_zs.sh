available_methods=("constant" "uniform")
folder_abs_path="${PWD}/bit_result/"
storage_path="${PWD}/ratio_result_zs"
cd ..
# exp1
export LP_BITS_THRESHOLD=3.0
model_name='opt'
model_size="1.3b"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $storage_path \
    -p $folder_abs_path --zeroshot --adafile -m "constant","uniform" 

export LP_BITS_THRESHOLD=6.0
model_name='bloom'
model_size="1b1"
bash test_plan_acc_cmd.sh --model_name $model_name --model_size $model_size --storage_path $storage_path \
    -p $folder_abs_path --zeroshot --adafile -m "constant","uniform" 