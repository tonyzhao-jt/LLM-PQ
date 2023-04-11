import os
import torch 
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import rpc
import numpy as np
import threading
from time import perf_counter

import logging
import copy 

from qllm.utils import to_dtype_recursive, to_device_recursive

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# configure formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# configure file handler
file_handler = logging.FileHandler('example.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

CMD_STOP = 0
CMD_SCHED = 1

def init_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# write a dist config class
from dataclasses import dataclass
@dataclass
class DistConfig:
    local_rank: int
    rank: int
    group_rank: int
    world_size: int
    ngpus: int

    def __init__(self, local_rank, rank, group_rank, world_size, ngpus):
        self.local_rank = local_rank
        self.rank = rank
        self.group_rank = group_rank
        self.world_size = world_size
        self.ngpus = ngpus

# utils
def to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, tuple):
        return tuple(to_device(t, device) for t in tensor)
    elif isinstance(tensor, list):
        return [to_device(t, device) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: to_device(v, device) for k, v in tensor.items()}
    else:
        return tensor
    
# dist / comm utils
def create_device_mesh(rank, local_rank, world_size):
    node_first_rank = rank - local_rank
    # get first_rank of each node and ngpus for each node
    # sort by first_rank, then we got the whole device mesh
    dist.init_process_group(backend='gloo', init_method='env://')

    node_info = torch.tensor([node_first_rank, local_rank], dtype=torch.int64)
    node_info_list = [torch.zeros(len(node_info), dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(node_info_list, node_info)

    # based on the first node, create a mesh with ranks has the same first rank on the row
    # and ranks has the same local rank on the column
    device_mesh = {}
    for i in range(world_size):
        first_rank, local_rank = node_info_list[i].tolist()
        if first_rank not in device_mesh:
            device_mesh[first_rank] = []
        device_mesh[first_rank].append(local_rank + first_rank)
    return device_mesh

def get_neighbor_ranks(device_mesh, rank):
    for first_rank, ranks in device_mesh.items():
        if rank in ranks:
            return ranks
def get_local_rank_by_device_mesh(device_mesh, rank):
    for first_rank, ranks in device_mesh.items():
        if rank in ranks:
            return rank - first_rank

def set_device_map(rank, device_mesh, schedules, rpc_options):
    stage_rank_order = list(schedules.keys())
    if rank not in stage_rank_order:
        return # only when rank is used
    i = stage_rank_order.index(rank)
    # determine the mapping from stage to device
    cur_stage_rank = stage_rank_order[i]
    next_stage_rank = stage_rank_order[(i + 1) % len(stage_rank_order)] # possible to be in the same node
    # cur_stage local rank
    cur_stage_local_rank = get_local_rank_by_device_mesh(device_mesh, cur_stage_rank)
    # next_stage local rank
    next_stage_local_rank = get_local_rank_by_device_mesh(device_mesh, next_stage_rank)
    if cur_stage_rank == next_stage_rank:
        pass
    else:
        logger.info(f"set device map for worker{cur_stage_rank}:{cur_stage_local_rank} to worker{next_stage_rank}:{next_stage_local_rank}")
        rpc_options.set_device_map(f"worker{next_stage_rank}", {cur_stage_local_rank: next_stage_local_rank})


def init_env():
    seed = 42
    init_random_seed(seed)
    ngpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    group_rank = int(os.environ['GROUP_RANK'])
    # neighbor ranks
    device_mesh = create_device_mesh(rank, local_rank, world_size)
    dist_cfg = DistConfig(local_rank, rank, group_rank, world_size, ngpus)
    return dist_cfg, device_mesh

# RPC CONTEXT
from typing import Any, Callable, List, Optional, Tuple, Type, Union
DistCmdHandler: Type = Callable[[int, Tuple[torch.Tensor, ...]], None]

class DistContext:
    """Parent class for distributed context managers."""

    def __init__(self, init_args: tuple, init_kwargs: dict):
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._world_size = init_kwargs['world_size']
        self._rank = init_kwargs['rank']
        self._initialized = False

    def init(self) -> None:
        """Initialize the distributed context."""
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the distributed context."""
        self._initialized = False

    def __enter__(self):
        assert not self._initialized
        self.init()
        return self

    def __exit__(self, *args):
        assert self._initialized
        self.shutdown()

class DistRpcContext(DistContext):
    """The singleton distributed RPC context manager."""

    def init(self) -> None:
        """Initialize the distributed context."""
        super().init()
        rpc.init_rpc(*self._init_args, **self._init_kwargs)

    def shutdown(self) -> None:
        """Wait for all RPCs to finish and shutdown the distributed context."""
        super().shutdown()
        rpc.shutdown()

    def cmd_broadcast(self, remote_cmd_handler: DistCmdHandler, cmd: int,
                      tensors: Optional[Tuple[torch.Tensor, ...]]=None) -> None:
        """Broadcast a command."""
        assert self._initialized
        if tensors is None:
            tensors = ()
        futs = []
        for rank in range(self._world_size):
            if rank != self._rank:
                fut = rpc.rpc_async(rank, remote_cmd_handler, args=(cmd, tensors))
                futs.append(fut)
        torch.futures.wait_all(futs)

# ideally, returns the pipline + quantization schedules
# key functions here
def get_shard_strategy(model_cpu):
    return [], [], []

# encode schedule to 1d list for communication
def encode_schedule(schedule):
    assert type(schedule) is dict, "the schedule should be a dict"
    stage_ranks = []
    stage_layers = []
    stage_qlvs = []
    for rank, cfg in schedule.items():
        layer_start = cfg['layers'][0]
        layer_end = cfg['layers'][-1]
        qlvs = cfg['bits']
        stage_ranks.append(rank)
        stage_layers.append(layer_start) 
        stage_layers.append(layer_end)
        for qlv in qlvs:
            stage_qlvs.append(qlv)
    return stage_ranks, stage_layers, stage_qlvs

# decode the schedule back for operation
def decode_schedule(stage_ranks, stage_layers, stage_qlvs):
    schedule = {}
    num_stages = len(stage_ranks)
    assert len(stage_layers) == 2 * num_stages, "the number of layers is not matched with the number of stages"
    passed_layers = 0
    for i in range(num_stages):
        rank = stage_ranks[i]
        layer_start = stage_layers[2 * i]
        layer_end = stage_layers[2 * i + 1]
        num_layers = layer_end - layer_start
        qlvs = stage_qlvs[passed_layers: passed_layers + num_layers]
        passed_layers = passed_layers + num_layers
        schedule[rank] = {'layers': [layer_start, layer_end], 'bits': qlvs}
    return schedule
    

def _verify_schedules(schedule, layers_num):
    # check the layer is start-end connected
    total_num_layers = 0
    for rank, cfg in schedule.items():
        layer_start = cfg['layers'][0]
        layer_end = cfg['layers'][-1]
        if rank == 0:
            former_layer = layer_end
        else:
            assert layer_start == former_layer, "the layer is not partitioned connected, pls check"
            former_layer = layer_end
        # check quantization is matched
        num_layers = layer_end - layer_start
        qlvs = cfg['bits']
        total_num_layers = total_num_layers + num_layers
        assert len(qlvs) == num_layers, "the number of quantization levels is not matched with the number of layers"
    
    assert total_num_layers == layers_num, "the number of layers is not matched with the model layers"
    logger.info("the schedule is verified")


class ThreadSafeCounter:
    """Thread-safe counter."""

    def __init__(self, value: int=0):
        self._value = value
        self._cond = threading.Condition()

    @property
    def value(self) -> int:
        """Current counter value."""
        with self._cond:
            val = self._value
            self._cond.notify_all()
        return val

    def add(self, quantity: int=1) -> None:
        """Add to counter atomically."""
        with self._cond:
            self._value += quantity
            self._cond.notify_all()

    def set(self, value: int=0) -> None:
        """Set (or reset) counter value."""
        with self._cond:
            self._value = value
            self._cond.notify_all()

    def wait_gte(self, threshold: int) -> None:
        """Wait until counter >= threshold."""
        with self._cond:
            while self._value < threshold:
                self._cond.wait()
results_counter = ThreadSafeCounter()

# hooks required for the pipeline
def forward_pre_hook_to_device(_module, inputs) \
    -> Union[Tuple[torch.tensor], Tuple[Tuple[torch.Tensor]]]:
    """Move tensors to the compute device (e.g., GPU), if needed."""
    assert isinstance(inputs, tuple)
    assert len(inputs) == 1
    if isinstance(inputs[0], torch.Tensor):
        inputs = (inputs,)
    tensors_dev = tuple(t.to(_module.device) for t in inputs[0])
    return tensors_dev if len(tensors_dev) == 1 else (tensors_dev,)

def forward_hook_to_cpu(_module, _inputs, outputs) -> Union[torch.tensor, Tuple[torch.Tensor]]:
    """Move tensors to the CPU, if needed."""
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    assert isinstance(outputs, tuple)
    print("outputs", outputs)
    tensors_cpu = tuple(t.cpu() for t in outputs)
    return tensors_cpu[0] if len(tensors_cpu) == 1 else tensors_cpu

# model factory
# We did three steps here
# 1. based on the partition rule, get the correct layers
# 2. put them to correct device
# 3. set the correct quantization level, load the correct weight (PS: Do later)
"""RPC communication module."""
import threading
class DistRpcPipelineStage:
    """Wrap a module that is not RPC-aware to manage threading and memory."""
    # NOTE: message ordering is NOT enforced!

    def __init__(self, module_cls: Type[nn.Module], module_args: Optional[tuple]=None,
                 module_kwargs: Optional[dict]=None, \
                 stage_id:int = None, local_rank:int=None, comm_type:str=None,):
        super().__init__()
        if module_args is None:
            module_args = ()
        if module_kwargs is None:
            module_kwargs = {}
        # _sem_fwd limits RPC threads in forward(), and thus data memory requirements (in + out).
        # _sem_mod limits Module thread parallelism, and thus processing memory requirements.
        # If each stage is configured for single-thread Module processing, then N=1.
        # Ideally, for _sem_mod value=N:
        # (1) N inputs are being or have been received (prior to forward() or waiting on _sem_mod)
        # (2) N inputs are processing (acquired _sem_mod)
        # (3) N outputs are sending or waiting to send (released _sem_mod)
        # More generally, however, the local stage may be backed up at any of these three steps,
        # depending on its performance relative to other stages and network conditions.
        self._sem_fwd = threading.Semaphore(value=3) # value = 3*N
        self._sem_mod = threading.Semaphore(value=1) # value = N
        # here should me a nn.Sequential Module, and a function that set quantization bit should be here
        # self._module = module_cls(*module_args, **module_kwargs)
        # TODO: replace with single node strategy
        self.is_master = stage_id==0
        self._module = module_cls
        self._next_rref = None
        self._results_to = None
        self._results_cb = None

        self.stage_id = stage_id
        self.comm_type = comm_type
        self.stage_device = torch.device(f"cuda:{local_rank}")
        self.module_to(self.stage_device)

    def module_to(self, *args, **kwargs) -> None:
        """Wrap the module's `nn.Module.to` method (`device` can be be a `str`)."""
        if self.is_master:
            self._module = self._module.to(*args, **kwargs)
        else:
            self._module.decoder_layers_to_device(*args, **kwargs)
        # print(self._module, args, kwargs)

    def set_next(self, stage_rref: rpc.RRef) -> None:
        """Set the RRef of the next pipeline stage - used by all stages except the last."""
        self._next_rref = stage_rref

    def set_results(self, results_to: Union[int, rpc.WorkerInfo, str],
                    results_cb: Callable[[Any], None]) -> None:
        """Set the results destination - used by only the last stage."""
        self._results_to = results_to
        self._results_cb = results_cb
        # print("set results", results_to, results_cb)

    def wait_for_ready(self) -> None:
        """Wait for this stage to be ready to receive data - MUST be called from previous stage."""
        # NOTE: This approach breaks down if the previous stage fails to send data afterward.
        self._sem_fwd.acquire() # pylint: disable=consider-using-with
        
    def __call__(self, inputs: Any) -> None:
        """Wrap the module's callable method."""
        inputs = to_device(inputs, self.stage_device)
        # print("on", self.stage_id)
        with self._sem_mod:
            outputs = self._module.decode(inputs)
        if self._next_rref is not None:
            # Sending must be asynchronous, otherwise we lose pipeline parallelism.
            # However, don't try to send until the next stage is ready.
            # If we were to initiate the async send (and then release _sem_fwd) too soon,
            # outbound data could get backlogged in this stage when the next stage is slow.
            self._next_rref.rpc_sync().wait_for_ready()
            self._next_rref.rpc_async().__call__(outputs)
        else:
            assert self._results_to is not None
            assert self._results_cb is not None
            # There's no synchronization with the results handler, just send the data.
            rpc.rpc_sync(self._results_to, self._results_cb, args=(outputs,))
        # Now release so that another microbatch may be received.
        self._sem_fwd.release()
    
    def test_fwd(self, input):
        # print("run", input)
        input = to_device(input, self.stage_device)
        outputs = self._module.decode(input)
        if self._next_rref:
            self._next_rref.rpc_async().test_fwd(outputs)
        else:
            assert self._results_to is not None
            assert self._results_cb is not None
            # There's no synchronization with the results handler, just send the data.
            rpc.rpc_sync(self._results_to, self._results_cb, args=(outputs,))

    def mod_register_buffer(self, *args, **kwargs) -> None:
        """Wrap the module's `register_buffer()` method."""
        return self._module.register_buffer(*args, **kwargs)

    def mod_register_forward_hook(self, *args, **kwargs) -> None:
        """Wrap the module's `register_forward_hook()` method."""
        return self._module.register_forward_hook(*args, **kwargs)

    def mod_register_forward_pre_hook(self, *args, **kwargs) -> None:
        """Wrap the module's `register_forward_pre_hook()` method."""
        return self._module.register_forward_pre_hook(*args, **kwargs)

class DistRpcPipeline:
    """A distributed RPC pipeline which links `DistRpcPipelineStage` RRefs."""

    def __init__(self, stage_rrefs: List[rpc.RRef], results_to: Union[int, rpc.WorkerInfo, str],
                 results_cb: Callable[[Any], None]):
        super().__init__()
        self._rref_list = stage_rrefs
        self._link_pipeline(results_to, results_cb)

    def rpc_register_buffer(self, name: str, tensors: List[Optional[torch.Tensor]],
                            **kwargs: dict) -> None:
        """Add buffers to RPC modules."""
        if len(tensors) != len(self._rref_list):
            raise ValueError(f"tensors length ({len(tensors)}) doesn't match pipeline length "
                             f"({len(self._rref_list)})")
        futs = [rref.rpc_async().mod_register_buffer(name, tensor, **kwargs)
                for rref, tensor in zip(self._rref_list, tensors)]
        torch.futures.wait_all(futs)

    # rpc_async returns a future object
    def rpc_register_forward_pre_hook(self, hook: Callable[..., None], first: bool=True) -> None:
        """Register forward pre hook."""
        rrefs = self._rref_list if first else self._rref_list[1:]
        hook_futures = [rref.rpc_async().mod_register_forward_pre_hook(hook) for rref in rrefs]
        torch.futures.wait_all(hook_futures)

    def rpc_register_forward_hook(self, hook: Callable[..., None], last: bool=True) -> None:
        """Register forward hook."""
        rrefs = self._rref_list if last else self._rref_list[:-1]
        hook_futures = [rref.rpc_async().mod_register_forward_hook(hook) for rref in rrefs]
        torch.futures.wait_all(hook_futures)

    def _link_pipeline(self, results_to, results_cb):
        n_stages = len(self._rref_list)
        futs = [self._rref_list[i].rpc_async().set_next(self._rref_list[i + 1])
                for i in range(n_stages - 1)]
        futs.append(self._rref_list[-1].rpc_async().set_results(results_to, results_cb))
        torch.futures.wait_all(futs)

        # [self._rref_list[i].rpc_sync().set_next(self._rref_list[i + 1])
        #         for i in range(n_stages - 1)]
        # self._rref_list[-1].rpc_sync().set_results(results_to, results_cb)

    def enqueue_tensor(self, tensor: torch.Tensor) -> None:
        """Insert data into the front of the pipeline."""
        self._rref_list[0].rpc_sync().wait_for_ready()
        self._rref_list[0].rpc_async().__call__(tensor)
        # self._rref_list[0].rpc_async().test_fwd(tensor.cpu()) # very important that the data sent should on the same device.

def _dist_rpc_pipeline_stage_factory(*args, **kwargs) -> DistRpcPipelineStage:
    """Get a `rpc.DistRpcPipelineStage` instance on the globally-configured `devices.DEVICE`."""
    stage = DistRpcPipelineStage(*args, **kwargs)
    return stage


def dist_rpc_pipeline_factory(model_cpu: nn.Module, sharding_strategy: dict, results_to: int,
                              results_cb: Callable[[Any], None]) -> DistRpcPipeline:
    """Get an RPC pipeline instance."""
    stage_rrefs = []
    stage_cnt = 0

    dst_rank_order = list(sharding_strategy.keys())
    stage_numbers = len(dst_rank_order)

    for dst_rank, stage_cfgs in sharding_strategy.items():
        print(f"Rank {dst_rank} module sharded")
        sharded_model = model_cpu.shard_model(sharding_strategy, dst_rank)
        neighbor_ranks = get_neighbor_ranks(device_mesh, dst_rank)
        if stage_cnt == stage_numbers - 1:
            next_rank = 0 # meet the last one
            comm_type = "cpu"
        else:
            next_rank = dst_rank_order[stage_cnt + 1]
            if next_rank in neighbor_ranks:
                comm_type = f"cuda:{neighbor_ranks.index(next_rank)}"
            else:
                comm_type = "cpu"
        dst_local_rank = neighbor_ranks.index(dst_rank)
        
        print(f"Stage {stage_cnt} on {dst_rank} with {comm_type} communication to {next_rank}, local rank {dst_local_rank}")

        rref = rpc.remote(dst_rank, _dist_rpc_pipeline_stage_factory, args=(sharded_model,),
                           kwargs={'stage_id': stage_cnt, 'module_kwargs': {}, \
                                   'local_rank':dst_local_rank, 'comm_type': comm_type})
        rpc.remote(dst_rank, logger.info,
                    args=("======= Stage %d on %d =======", stage_cnt, dst_rank))
        stage_cnt += 1
        stage_rrefs.append(rref)

    logger.info("results to rank %d", results_to)
    logger.info("results callback %s", results_cb.__name__ if results_cb else "None")
    logger.info("======= Pipeline created =======")
    logger.info("with %d stages", len(stage_rrefs))
    return DistRpcPipeline(stage_rrefs, results_to, results_cb)

import queue
sched_q = queue.Queue()
stop_event = threading.Event()
def handle_cmd(cmd: int, tensors: Tuple[torch.Tensor, ...]) -> None:
    """Process received commands."""
    if cmd == CMD_STOP:
        logger.info("handle_cmd: stop")
        stop_event.set()
    elif cmd == CMD_SCHED:
        logger.info("handle_cmd: sched")
        sched_q.put(tuple(t.tolist() for t in tensors))
    else:
        logger.warning("handle_cmd: Unknown command: %s", cmd)

final_result = {}
request_input_ids = {}
request_logit_processor = {}
request_loop_counter = {}
def handle_results(final_intermediate_result) -> None:
    request_id = final_intermediate_result[-1]
    request_loop_counter[request_id] += 1
    results_counter.add(1)
    # get original input id
    input_ids = request_input_ids[request_id]
    logits_processor = request_logit_processor[request_id]
    # generate new tokens
    outputs = model_pre_and_post.postprocess(final_intermediate_result, None)
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if request_loop_counter[request_id] < num_tokens_to_generate:
        request_input_ids[request_id] = new_input_ids
        request_token = model_pre_and_post.preprocess(new_input_ids, use_cache=True, request_id=request_id)
        logger.info(f"Request id {request_id} done for token {request_loop_counter[request_id]}")
        master_pipeline.enqueue_tensor(to_device_recursive(request_token, 'cpu'))


master_pipeline = None
def run_pipeline_rpc(model_cpu:list, tokenizer, dist_cfg: DistConfig, chunk:int=1, sharding_strategy=None) -> None:
    global master_pipeline
    """Run the pipeline using RPC communication."""
    rpc_opts = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128, rpc_timeout=60)
    rank = dist_cfg.rank
    data_rank = 0 # by default, use rank 0 as the data rank
    # verify the scheduling is ok to be set
    if rank == 0:
        if not sharding_strategy:
            sharding_strategy = get_shard_strategy(model_cpu)
        else:
            model_cpu._verify_shard_strategy(sharding_strategy)  
    set_device_map(rank, device_mesh, sharding_strategy, rpc_opts)
    # based on the schedule and device_mesh, determines the communication type
    with DistRpcContext((f"worker{rank}",),
                        { 'world_size': dist_cfg.world_size,
                          'rank': rank,
                          'rpc_backend_options': rpc_opts}
                       ) as dist_ctx:
        # Send or receive the schedule
        # if rank == 0:
        #     # the stage_layers, stage_quant, stage_ranks
        #     if not sharding_strategy:
        #         sharding_strategy = get_shard_strategy(model_cpu)
        #     else:
        #         layers_num = len(model_cpu)
        #         _verify_schedules(sharding_strategy, layers_num)
        #     stage_layers, stage_quant, stage_ranks = encode_schedule(sharding_strategy)
        #     # broad cast the scheduling to all workers
        #     logger.info("Scheduling: data rank: %s", data_rank)
        #     logger.info("Broadcasting schedule")
        #     dist_ctx.cmd_broadcast(handle_cmd, CMD_SCHED,
        #                            (torch.tensor(stage_layers),
        #                             torch.tensor(stage_quant),
        #                             torch.tensor(stage_ranks),
        #                             torch.tensor(data_rank)))
        #     # stage_schedules = sharding_strategy[0]
        # else:
        #     logger.info("Waiting for schedule")
        #     stage_layers, stage_quant, stage_ranks, data_rank = sched_q.get()
        #     pipe_schedules = decode_schedule(stage_layers, stage_quant, stage_ranks)
        #     if rank in pipe_schedules:
        #         stage_schedule = pipe_schedules[rank]
        #         logger.info("Stage %d schedule:", rank)
        #         logger.info(stage_schedule)
            # logger.info("Stage layers: %s", stage_layers)
            # logger.info("Stage quant: %s", stage_quant)
            # logger.info("Stage ranks: %s", stage_ranks)
            # logger.info("Data rank: %s", data_rank)
        
        if rank == 0: # master, process some data
            # create pipeline
            pipeline = dist_rpc_pipeline_factory(model_cpu, sharding_strategy, rank, handle_results)
            master_pipeline = pipeline
            # prepare test data
            def prepare_input(p_input_ids, request_id):
                generation_config = model_pre_and_post.generation_config
                request_token = model_pre_and_post.preprocess(p_input_ids, use_cache=True, request_id=request_id)
                inputs_tensor, model_input_name, model_kwargs = model_pre_and_post._prepare_model_inputs(
                    p_input_ids, generation_config.bos_token_id, {}
                )
                input_ids_seq_length = p_input_ids.shape[-1]
                logits_processor = LogitsProcessorList()
                # 8. prepare distribution pre_processing samplers
                logits_processor = model_pre_and_post._get_logits_processor(
                    generation_config=generation_config,
                    input_ids_seq_length=input_ids_seq_length,
                    encoder_input_ids=inputs_tensor,
                    prefix_allowed_tokens_fn=None,
                    logits_processor=logits_processor,
                )

                final_result[request_id] = None
                request_input_ids[request_id] = p_input_ids
                request_logit_processor[request_id] = logits_processor
                request_loop_counter[request_id] = 0

                return to_device_recursive(request_token, 'cpu')
            
            data_chunks = []
            for i in range(request_numbers):
                request_token = prepare_input(tokenizer.encode(fetch_prompts(bs_token, prompt_length), return_tensors="pt").cuda(), request_id=i)
                data_chunks.append(request_token)
            batch_size = len(data_chunks)
            print("pipeline")
            # fake the data chunks for test
            # TODO: make it real chunks
            # data_chunks = torch.chunk(sample_data_batch, chunk, dim=0)
            # other_decode_params = master_model.other_decode_params
            # dist_ctx.cmd_broadcast(handle_cmd, CMD_SCHED, other_decode_params)
        else:
            print("worker")
            # communicate to get other_decode_params
            # logger.info("Waiting for other params")
            # other_decode_params = sched_q.get()
            # assign the decode params to the corresponding refs

        
        if rank == data_rank:
            # pipeline.rpc_register_forward_hook(forward_hook_to_cpu)
            # pipeline.rpc_register_forward_pre_hook(forward_pre_hook_to_device)
            tik_data = perf_counter()
            # start results monitoring - see comments in handle_results
            # this call is asynchronous - wait for results to get end-to-end timings
            logger.info("start pipe data")
            start_count = results_counter.value
            # this only launch the tasks but not actually finish the tasks.
            for data_chunk in data_chunks:
                pipeline.enqueue_tensor(data_chunk)
            results_counter.wait_gte(start_count + len(data_chunks) * num_tokens_to_generate)
            tok_data = perf_counter()
            latency = tok_data - tik_data
            throughput = batch_size / latency
            logger.info("Latency is %f, throughput is %f", latency, throughput)

            for request_id, input_id_finally in request_input_ids.items():
                ouput_token = tokenizer.batch_decode(input_id_finally, skip_special_tokens=True)
                print(f"request {request_id} output token {ouput_token}")


from qllm.models.OPT import OPTForCausalLMSeq
from transformers import AutoTokenizer
from transformers import LogitsProcessorList, StoppingCriteriaList
from samples import fetch_prompts
if __name__ == '__main__':
    # load the LLM from QLLM
    # QLLM support the sequential execution of the decoders
    loaded_llm_cpu = OPTForCausalLMSeq.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
    model_pre_and_post = model_pre_and_post.cuda()

    # control the token generation
    num_tokens_to_generate = 50
    prompt_length = 512
    bs_token = 16 # how many sentence in a batch
    request_numbers = 2 # how many requests

    # print("decoder layernum", loaded_llm_cpu.model.decoder.get_decoder_layer_num())
    dist_cfg, device_mesh = init_env()

    # the sharding strategy basically like
    # Rank: {decoder_layer_idx: {"shard": [shard_idx, ...], "bits":[qbit, ...]}}
    sharding_strategy = {
        0: {},
        1: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
            2: {'shard': [0, 1], 'bits': [16, 16]},
            3: {'shard': [0, 1], 'bits': [16, 16]},
            4: {'shard': [0, 1], 'bits': [16, 16]},
            5: {'shard': [0, 1], 'bits': [16, 8]},
            6: {'shard': [0, 1], 'bits': [16, 16]},
            7: {'shard': [0, 1], 'bits': [16, 16]},
            8: {'shard': [0], 'bits': [16]},
        },
        8: {
            8: {'shard': [1], 'bits': [16]},
            9: {'shard': [0,1], 'bits': [16, 16]},
            10: {'shard': [0,1], 'bits': [8, 16]},
            11: {'shard': [0,1], 'bits': [16, 16]},
            # 350M
            12: {'shard': [0,1], 'bits': [16, 16]},
            13: {'shard': [0,1], 'bits': [16, 16]},
            14: {'shard': [0,1], 'bits': [8, 16]},
            15: {'shard': [0,1], 'bits': [16, 16]},
            16: {'shard': [0,1], 'bits': [16, 16]},
            17: {'shard': [0,1], 'bits': [16, 8]},
        },
        9:{
            18: {'shard': [0,1], 'bits': [16, 16]},
            19: {'shard': [0,1], 'bits': [16, 16]},
            20: {'shard': [0,1], 'bits': [8, 16]},
            21: {'shard': [0,1], 'bits': [16, 16]},
            22: {'shard': [0,1], 'bits': [16, 16]}, 
            23: {'shard': [0,1], 'bits': [16, 16]},
        }
    }
    # print(device_mesh)
    run_pipeline_rpc(loaded_llm_cpu, tokenizer, dist_cfg, chunk=2, sharding_strategy=sharding_strategy)