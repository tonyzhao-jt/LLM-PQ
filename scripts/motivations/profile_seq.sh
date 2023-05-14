# prefill stage
python3 profile_lat_gen.py --batch-size 2 --past-seq-length 0 --generated-seq-length 1 --input-seq-length 128 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 2 --past-seq-length 0 --generated-seq-length 1 --input-seq-length 512 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 4 --past-seq-length 0 --generated-seq-length 1 --input-seq-length 128 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 4 --past-seq-length 0 --generated-seq-length 1 --input-seq-length 512 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 8 --past-seq-length 0 --generated-seq-length 1 --input-seq-length 128 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 8 --past-seq-length 0 --generated-seq-length 1 --input-seq-length 512 --repeat 10 --step 50

# python3 profile_lat_gen.py --batch-size 4 --past-seq-length 0 --generated-seq-length 1 --input-seq-length 512 --repeat 10 --model-size 66b --step 50

# decoding stage
python3 profile_lat_gen.py --batch-size 2 --past-seq-length 512 --generated-seq-length 200 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 2 --past-seq-length 128 --generated-seq-length 200 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 4 --past-seq-length 512 --generated-seq-length 200 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 4 --past-seq-length 128 --generated-seq-length 200 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 8 --past-seq-length 512 --generated-seq-length 200 --repeat 10 --step 50
python3 profile_lat_gen.py --batch-size 8 --past-seq-length 128 --generated-seq-length 200 --repeat 10 --step 50