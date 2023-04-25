# python3 pipeedge_algo.py --adaptive
# 30b test
python adaptive_bits.py --model_size "30b" --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 2 1 
python qpipe_ilp.py --model_size "30b" --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 2 1 
python qpipe_ilp.py --model_size "30b" --SLO-aware --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 2 1 
python uniform.py --model_size "30b" --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 2 1 
python pipeedge_algo.py  --model_size "30b" --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 2 1 
python pipeedge_algo.py  --model_size "30b"  --adaptive --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 2 1 
python convert_and_valid_strat.py --model_size "30b" --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 2 1

# 66b
python adaptive_bits.py --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python qpipe_ilp.py --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python qpipe_ilp.py --model_size "66b" --SLO-aware --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python uniform.py --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python pipeedge_algo.py  --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python pipeedge_algo.py  --model_size "66b"  --adaptive --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2 
python convert_and_valid_strat.py --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2

# 175b test
python adaptive_bits.py --model_size "175b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 4
python qpipe_ilp.py --model_size "175b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 4
python qpipe_ilp.py --model_size "175b" --SLO-aware --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 4 
python uniform.py --model_size "175b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 4
python pipeedge_algo.py  --model_size "175b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 4
python pipeedge_algo.py  --model_size "175b"  --adaptive --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 4 
python convert_and_valid_strat.py --model_size "175b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 4