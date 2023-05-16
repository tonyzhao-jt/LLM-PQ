# compare different approaches of indicator's cost
folder_path = "/workspace/qpipe/3rd_party/gptq/zeroShot"
from utils import simple_model_info_parser
from gen_ind import generate_indicator
from gen_ind_hess import generate_indicator_hess
from time import perf_counter
args = simple_model_info_parser()
model_size = args.model_size
model_name = args.model_name


# our case
repeat = 3
dur_sum = 0
start = perf_counter()
for i in range(repeat):
    omega, dur = generate_indicator(model_name, model_size, folder_path)
    dur_sum += dur
end = perf_counter()
dur = (end - start + dur_sum) / repeat 
print(f"our case: {dur}")
# random.
# no cost.
print("random:", 0 )
# hessian
start = perf_counter()
for i in range(repeat):
    omega, dur = generate_indicator_hess(model_name, model_size, folder_path)
    dur_sum += dur
end = perf_counter()
dur = (end - start + dur_sum) / repeat
print(f"hessian: {dur}")

