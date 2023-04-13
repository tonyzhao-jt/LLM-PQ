from qpipe.profiler import profile_comm
import torch.distributed as dist
import pickle
import os 
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
    local_rank = os.environ['LOCAL_RANK']
    if local_rank == '0':
        # check if the comm_cost_model folder exists
        if not os.path.exists('comm_cost_model'):
            os.mkdir('comm_cost_model')
    dist.barrier()
    dataset = test_comm_speed()
    cost_model = profile_comm.fit_cost_model(dataset)

    # save cost_model into folder
    file_name = f"cost_model_{rank}.pkl"
    file_path = os.path.join('comm_cost_model', file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(cost_model, f)