import torch.distributed as dist
import pickle
import os 

from llm_pq.rpc import init_env_gloo, init_env
from llm_pq.profiler import profile_comm
import llm_pq
from llm_pq.p2p import (
    new_nccl_group
)

import argparse
def test_comm_speed():
    data_size_buffer, time_buffer = profile_comm.generate_cost_model_dataset(batch_size=16, hidden_space=2048, sample_num=15, warmup=3)
    rank = dist.get_rank()
    print(f"Rank {rank} communication times:")
    for pair, times in time_buffer.items():
        print(f"{pair}: {times.mean():.6f}s")
    return data_size_buffer, time_buffer

def parse_args():
    parser = argparse.ArgumentParser(description='LLM-PQ-CommProf')
    parser.add_argument('--nccl', action='store_true', help='use nccl backend')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    # new a nccl group
    if args.nccl:
        init_env()
        # get group
        nccl_group = new_nccl_group()
        rank = dist.get_rank()
        local_rank = os.environ['LOCAL_RANK']
        llm_pq._globals.__PIPELINE__MODEL__PARALLEL__GROUP__ = nccl_group
        llm_pq._globals.__DEVICE__INDEX__ = local_rank
    else:
        init_env_gloo()
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
    # check if tmp folder exists
    file_path = os.path.join('tmp', file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(cost_model, f)