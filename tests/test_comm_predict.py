from shaq.cost_model import CommCostModel
from shaq.utils import get_size_cpu
comm_cost_model = CommCostModel(comm_cost_model_folder='/workspace/qpipe/scripts/')
comm_cost_model.print_model_available_keys()

# generate a random tensor to test communication
import torch
b, t, c = 4, 1, 4096
x = torch.randn(b, t, c)
comm_cost = comm_cost_model.predict_comm_time(start_rank=0, end_rank=1, data_size=get_size_cpu(x, unit='MB'))
print(f"Communication cost: {comm_cost:.6f}s")