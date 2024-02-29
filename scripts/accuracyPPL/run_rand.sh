# motivation experiment
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 3 --task piqa,arc_easy,lambada 2>&1 | tee mot_opt_3.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 3 --rand-bit --task piqa,arc_easy,lambada 2>&1 | tee mot_opt_3-4.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 4 --task piqa,arc_easy,lambada 2>&1 | tee mot_opt_4.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 4 --rand-bit --task piqa,arc_easy,lambada 2>&1 | tee mot_opt_4-8.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 8 --task piqa,arc_easy,lambada 2>&1 | tee mot_opt_8.txt

CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-3b c4 --wbits 3 --task piqa,arc_easy,lambada 2>&1 | tee mot_bloom_3.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-3b c4 --wbits 3 --rand-bit --task piqa,arc_easy,lambada 2>&1 | tee mot_bloom_3-4.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-3b c4 --wbits 4 --task piqa,arc_easy,lambada 2>&1 | tee mot_bloom_4.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-3b c4 --wbits 4 --rand-bit --task piqa,arc_easy,lambada 2>&1 | tee mot_bloom_4-8.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-3b c4 --wbits 8 --task piqa,arc_easy,lambada 2>&1 | tee mot_bloom_8.txt