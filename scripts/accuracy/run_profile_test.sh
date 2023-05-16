gptq_zs_folder="/workspace/qpipe/3rd_party/gptq/zeroShot/"
cd $gptq_zs_folder
model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
# test
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 4 --task piqa --profile
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 2 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 3 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 4 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 8 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 18 --task piqa