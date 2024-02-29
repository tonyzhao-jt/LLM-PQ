from llm_pq.profiler import profile_comm
import torch.distributed as dist
from llm_pq.rpc import init_env
def test_comm_speed():
    data_size_buffer, time_buffer = profile_comm.generate_cost_model_dataset(batch_size=4, hidden_space=4096, sample_num=10, warmup=3)
    rank = dist.get_rank()
    print(f"Rank {rank} communication times:")
    for pair, times in time_buffer.items():
        print(f"{pair}: {times.mean():.6f}s")
    return data_size_buffer, time_buffer

if __name__ == "__main__":
    init_env()
    dataset = test_comm_speed()
    if dist.get_rank() == 0:
        cost_model = profile_comm.fit_cost_model(dataset)
    
    # save cost_model
    if dist.get_rank() == 0:
        import pickle
        with open("cost_model.pkl", "wb") as f:
            pickle.dump(cost_model, f)