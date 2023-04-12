from qpipe.profiler import profile_comm
import torch.distributed as dist
import pickle
from qpipe.rpc import init_env
def test_comm_speed():
    data_size_buffer, time_buffer = profile_comm.generate_cost_model_dataset(batch_size=4, hidden_space=4096, sample_num=10, warmup=3)
    rank = dist.get_rank()
    print(f"Rank {rank} communication times:")
    for pair, times in time_buffer.items():
        print(f"{pair}: {times.mean():.6f}s")
    return data_size_buffer, time_buffer

if __name__ == "__main__":
    init_env()
    rank = dist.get_rank()
    dataset = test_comm_speed()
    cost_model = profile_comm.fit_cost_model(dataset)
    # save cost_model
    with open(f"cost_model_{rank}.pkl", "wb") as f:
        pickle.dump(cost_model, f)