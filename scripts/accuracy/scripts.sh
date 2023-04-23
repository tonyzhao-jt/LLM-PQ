# gptq provided scripts
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-125m c4 --wbits 4 --task piqa

CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py facebook/opt-1.3b c4 --wbits 4 --task piqa

CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --task piqa
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --wbits 4 --task piqa

CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --task piqa --batch-size 4
CUDA_VISIBLE_DEVICES=0 python3 main.py bigscience/bloom-560m c4 --wbits 4 --task piqa --batch_size 4

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