# gen hess
# for test
# python3 gen_ind_hess.py --model-name opt --model-size 125m 

# experiment setups
# python3 gen_ind_hess.py --model-name opt --model-size 30b 
# duration: 15670.865950480918
# python3 gen_ind_hess.py --model-name opt --model-size 66b 
# duration: 25282.890390469693

# for test
python3 gen_ind.py --model-name opt --model-size 125m 
# experiment setups
# python3 gen_ind.py --model-name opt --model-size 13b 
# python3 gen_ind.py --model-name opt --model-size 30b
# duration: 215.59935343964025
# python3 gen_ind.py --model-name opt --model-size 66b 
# duration: 434.7788
# python3 gen_ind.py --model-name bloom --model-size 176b 
