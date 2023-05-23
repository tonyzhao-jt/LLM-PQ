import time
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Any
from torch import Tensor
import numpy as np
from ..utils import get_size_cpu

def build_dummy(batch_size: int, hidden_space: int, sample_num: int) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    '''
    build dumpy data for profiling communication cost of each worker (device) pair
    '''
    send_dummy: Dict[int,List[Tensor]] = {}
    recv_dummy: Dict[int,List[Tensor]] = {}
    for i in range(dist.get_world_size()):
        for j in range(batch_size):
            data = [torch.rand(j, 512, hidden_space + 1024 * i) for i in range(sample_num)] \
                + [torch.rand(j, 128, hidden_space + 1024 * i) for i in range(sample_num)] \
                + [torch.rand(j, 1, hidden_space + 1024 * i) for i in range(sample_num)]
        send_dummy[i] = data
        recv_dummy[i] = [torch.zeros_like(data[i]) for i in range(sample_num)]
    return send_dummy, recv_dummy

def generate_sender(send_dumpy: Dict[int,List[Tensor]], warmup:int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[float]]]:
    '''
    generate the datatset for cost model of each worker pair. dataset: (data_size, time)
    '''
    rank, world_size = dist.get_rank(), dist.get_world_size()
    data_size_buffer: Dict[str, List[Tensor]] = {}
    time_buffer: Dict[str, List[float]] = {}
    # generate sending dataset for device pairs whose src device is rank
    for i in range(world_size):
        if i != rank:
            data_size_buffer[f'{rank}_{i}'] = []
            time_buffer[f'{rank}_{i}'] = []
            dummy_data = send_dumpy[i]
            for n in range(len(dummy_data)):
                avg_time = []
                for epoch in range(1, 3 * warmup):
                    start = time.time()
                    dist.send(dummy_data[n], i, tag=0)
                    end = time.time()
                    if epoch > warmup:
                        avg_time.append(end - start)
                        print("time")
                data_size_buffer[f'{rank}_{i}'].append(get_size_cpu(dummy_data[n], unit='MB')) # MB
                time_buffer[f'{rank}_{i}'].append(sum(1000 * avg_time) / len(avg_time)) # ms
            data_size_buffer[f'{rank}_{i}'] = torch.tensor(data_size_buffer[f'{rank}_{i}'])
            time_buffer[f'{rank}_{i}'] = torch.tensor(time_buffer[f'{rank}_{i}'])
    # sync
    dist.barrier()
    return data_size_buffer, time_buffer


def generate_receiver(dumpy_data: List[Tensor], sender_rank: int, warmup: int):
    '''
    waiting for receiving dummy data from the sender.
    '''
    for n in range(len(dumpy_data)):
        for _ in range(1, 3 * warmup):
            dist.recv(dumpy_data[n], sender_rank, tag=0)
    # sync
    dist.barrier()

def generate_cost_model_dataset(batch_size: int, hidden_space: int, sample_num: int, warmup: int) -> Tuple[Dict[str, List[Any]], Dict[str, List[float]]]:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    send_dummy, recv_dummy = build_dummy(batch_size, hidden_space, sample_num)

    for sender in range(world_size):
        if sender == rank:
            data_size_buffer, time_buffer = generate_sender(send_dummy, warmup)
        else:
            generate_receiver(recv_dummy[sender], sender, warmup)

    return data_size_buffer, time_buffer

def fit_cost_model(dataset: Tuple[Dict[str, List[Tensor]], Dict[str, List[float]]]) -> Dict[str, np.ndarray]:
    '''
    fit the cost model for each worker pair
    '''
    cost_model: Dict[str, np.ndarray] = {}
    data_size_buffer, time_buffer = dataset

    for pair_key in data_size_buffer.keys():
        model = np.polyfit(data_size_buffer[pair_key], time_buffer[pair_key], 1) # alpha-beta model (y = alpha * x + beta)
        cost_model[pair_key] = model
    return cost_model



