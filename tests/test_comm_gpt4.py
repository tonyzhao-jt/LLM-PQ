import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import argparse

import os
import time


def empty(*args, **kwargs):
    pass

def measure(tensor_size, comm_mode):
    x = torch.randn(tensor_size, 768).cuda(0) if comm_mode == "cuda" else torch.randn(tensor_size, 768)

    tik = time.time()
    for _ in range(10):
        fut = rpc.rpc_async("worker1", empty, args=(x, ))
        res = fut.wait()
    if comm_mode == "cuda":
        torch.cuda.current_stream("cuda:0").synchronize()
    tok = time.time()
    print(f"{comm_mode} RPC with tensor size {tensor_size} total execution time: {tok - tik}")

def run_worker(rank, tensor_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        options.set_device_map("worker1", {0: 1})
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=2,
            rpc_backend_options=options
        )
        measure(tensor_size, comm_mode="cpu")
        measure(tensor_size, comm_mode="cuda")
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=2,
            rpc_backend_options=options
        )

    rpc.shutdown()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_size', type=int, default=1000, help='Specify the tensor size to be tested')
    args = parser.parse_args()

    world_size = 2
    mp.spawn(run_worker, args=(args.tensor_size,), nprocs=world_size, join=True)
