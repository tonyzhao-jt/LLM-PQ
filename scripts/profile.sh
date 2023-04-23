# prompts much larger than the generated tokens
export CUDA_VISIBLE_DEVICES=3 # use last one

python3 profile_lat.py --batch-size 16 --past-seq-length 512 --generated-seq-length 100 --repeat 10
python3 profile_lat.py --batch-size 8 --past-seq-length 512 --generated-seq-length 100 --repeat 10 
python3 profile_lat.py --batch-size 4 --past-seq-length 512 --generated-seq-length 100 --repeat 10 

python3 profile_lat.py --batch-size 16 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-size 66b
python3 profile_lat.py --batch-size 8 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-size 66b
python3 profile_lat.py --batch-size 4 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-size 66b

python3 profile_lat.py --batch-size 16 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-size 30b
python3 profile_lat.py --batch-size 8 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-size 30b
python3 profile_lat.py --batch-size 4 --past-seq-length 512 --generated-seq-length 100 --repeat 10 --model-size 30b