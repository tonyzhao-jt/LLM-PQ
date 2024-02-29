# export TEST_RATIO=0.5
# bash run_mot_pp.sh
# bash run_mot_zs.sh
# export TEST_RATIO=0.3
# bash run_mot_pp.sh
# bash run_mot_zs.sh
model_storage_path='/mnt/bn/zjtnaslq/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
bash run_mot_pp.sh
bash run_mot_zs.sh