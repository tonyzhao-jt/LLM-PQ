# gptq provided scripts
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 4 --task piqa



CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 4 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 4 --task piqa \
--ada-file "/workspace/qpipe/scripts/accuracy/bit_for_gptq_test/rand_opt_1.3b_bit_ass.pkl"

CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --wbits 4 --task piqa

CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --wbits 4 --task piqa \
--ada-file '/workspace/qpipe/scripts/accuracy/bit_for_gptq_test/rand_bloom_560m_bit_ass.pkl'

CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-30b c4 --wbits 4 --task piqa \
--ada-file '/workspace/qpipe/scripts/accuracy/bit_for_gptq_test/qpipe_30b_Tesla_T4_2_Tesla_V100-SXM2-32GB_1_acc_test.pkl'


CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --task piqa --batch-size 4
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --wbits 4 --task piqa --batch_size 4


CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 4 --task piqa --profile 
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --wbits 4 --task piqa --profile

CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 4 --task piqa arc lambada --profile
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-3b c4 --wbits 4 --task piqa arc lambada --profile




# TASK_REGISTRY = {
#     "lambada": LAMBADA,
#     "piqa": piqa.PiQA,
#     "arc_easy": arc.ARCEasy,
#     "arc_challenge": arc.ARCChallenge,
#     "boolq": superglue.BoolQ,
#     "cb": superglue.CommitmentBank,
#     "copa": superglue.Copa,
#     "wic": superglue.WordsInContext,
#     "multirc": superglue.MultiRC,
#     "rte": glue.RTE,
#     "record": superglue.ReCoRD,
#     "wsc": superglue.SGWinogradSchemaChallenge,
#     "storycloze": storycloze.StoryCloze2018
# }

# in paper, use dataset and lambda 'wikitext2', 'ptb', 'c4'