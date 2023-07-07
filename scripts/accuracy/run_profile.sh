gptq_zs_folder="/workspace/qpipe/3rd_party/gptq/zeroShot/"
cd $gptq_zs_folder
model_storage_path='/data/llms/'
export TRANSFORMERS_CACHE=$model_storage_path
export CUDA_VISIBLE_DEVICES=3
# test
# python3 main.py facebook/opt-125m c4 --wbits 8 --task piqa --profile-hessian
# python3 main.py facebook/opt-125m c4 --wbits 8 --task piqa --profile
# all indicators required.
# python3 main.py facebook/opt-13b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# python3 main.py facebook/opt-30b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# python3 main.py facebook/opt-66b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# python3 main.py bigscience/bloom c4 --wbits 4 --task piqa,arc_easy,lambada --profile
# python3 main.py facebook/opt-175b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
for bit in 8
do
    echo "Try collect Hess"
    # python3 main.py facebook/opt-13b c4 --wbits ${bit} --task piqa
    python3 main.py facebook/opt-30b c4 --wbits ${bit} --task piqa
    python3 main.py facebook/opt-66b c4 --wbits ${bit} --task piqa
    # python3 main.py facebook/bloom c4 --wbits ${bit} --task piqa
done