export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_CACHE='/data/llms/'
# CPU Not Enough
export LOAD_IN_NP="1"
# sample
# python3 weight_convert_numpy.py --model-size 125m
# all used
# python3 weight_convert_numpy.py --model-size 13b
# python3 weight_convert_numpy.py --model-size 30b
# python3 weight_convert_numpy.py --model-size 66b
python3 weight_convert_numpy.py --model-name bloom --model-size 176b
