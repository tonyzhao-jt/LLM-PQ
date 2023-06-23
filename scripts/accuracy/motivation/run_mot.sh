# export CUDA_VISIBLE_DEVICES=1
export TEST_RATIO=0.6
bash run_mot_pp.sh
bash run_mot_zs.sh
export TEST_RATIO=0.3
bash run_mot_pp.sh
bash run_mot_zs.sh