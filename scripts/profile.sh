# prompts much larger than the generated tokens
python3 profile_lat.py --batch-size 16 --past-seq-length 512 --generated-seq-length 512 --step 32