CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-30b c4 --wbits 4 --task piqa --profile
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-1b1 c4 --wbits 4 --task piqa --profile