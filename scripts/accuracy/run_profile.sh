gptq_zs_folder="/workspace/qpipe/3rd_party/gptq/zeroShot/"
cd $gptq_zs_folder
# test
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 4 --task piqa,arc_easy,lambada --profile



CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-13b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-30b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-66b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-175b c4 --wbits 4 --task piqa,arc_easy,lambada --profile
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-176b c4 --wbits 4 --task piqa,arc_easy,lambada --profile