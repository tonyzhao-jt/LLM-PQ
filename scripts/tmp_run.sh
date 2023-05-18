python adaptive_bits.py --model_size "66b" --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 4 2
python qpipe_ilp.py --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python qpipe_ilp.py --model_size "66b" --SLO-aware --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python uniform.py --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python pipeedge_algo.py  --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python pipeedge_algo.py  --model_size "66b"  --adaptive --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2 
python convert_and_valid_strat.py --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python adaqpipe.py --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python pipeedge_ilp.py  --model_size "66b" --device_names Tesla_V100-SXM2-32GB NVIDIA_A100-SXM4-40GB --device_numbers 4 2
python pipeedge_adaptive.py  --model_size "66b" --device_names Tesla_T4 Tesla_V100-SXM2-32GB --device_numbers 4 2


