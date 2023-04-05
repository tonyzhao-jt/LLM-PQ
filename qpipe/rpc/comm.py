import logging
from qpipe.logger import logger
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


import queue
sched_q = queue.Queue()
stop_event = threading.Event()
CMD_STOP = 0
CMD_SCHED = 1
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