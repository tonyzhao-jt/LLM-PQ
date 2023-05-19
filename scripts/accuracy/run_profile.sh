gptq_zs_folder="/workspace/qpipe/3rd_party/gptq/zeroShot/"
cd $gptq_zs_folder
model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
# test
# CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# all indicators required.
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-13b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-30b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-66b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-175b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
for bit in 2 3 4 8
do
    # CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-13b c4 --wbits ${bit} --task piqa
    # CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-30b c4 --wbits ${bit} --task piqa
    # CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-66b c4 --wbits ${bit} --task piqa
    # CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/bloom c4 --wbits ${bit} --task piqa
done