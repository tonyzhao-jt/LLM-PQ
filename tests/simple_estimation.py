from qllm.utils import ModelMemEstimator
from qllm.models import opt 
opt_125M, tokenizer = opt.get_empty_model()
h2, h1 = opt_125M.model.decoder.layers[0].fc1.weight.shape
b = 1
s = 1 # sentence length
n = 1 # generated tokens

print(h1, h2)
model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n)
print("Estimated Hidden Space size", model_mem_estimator.estimate_hidden_space())