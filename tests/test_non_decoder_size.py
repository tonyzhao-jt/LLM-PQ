
import os
import torch 
from torch.distributed import rpc 

from time import perf_counter


from transformers import AutoTokenizer
from transformers import LogitsProcessorList, StoppingCriteriaList

from qllm.models.OPT import OPTForCausalLMSeq
from qllm.utils import (
    to_device_recursive,
    get_iter_variable_size,
)

from qpipe import (
    init_random_seed,
    fetch_prompts
)

from qpipe.rpc import (
    init_env, DistConfig, set_device_map,
    DistRpcContext
)
from qpipe.logger import logger
from qpipe.thread import ThreadSafeCounter
from qpipe.partitioner import get_shard_strategy
from qpipe.pipe import (
    dist_rpc_pipeline_factory
)

import lptorch
import pickle
from qllm.models.OPT.opt import model_cards
from qpipe.utils import get_cuda_occupation_by_command
if __name__ == '__main__':

    # load the LLM from QLLM
    # with weight
    # loaded_llm_cpu = OPTForCausalLMSeq.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    # wo weight
    # os.environ["NCCL_SOCKET_IFNAME"] = "enp225s0"
    # os.environ["GLOO_SOCKET_IFNAME"] = "enp225s0"
    # export GLOO_SOCKET_IFNAME=enp225s0
    os.environ['SET_DECODERS_META'] = "1"

    model_size = "175b"
    config = model_cards[model_size]
    loaded_llm_cpu = OPTForCausalLMSeq._from_config(config, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b")

    pipeline_strategy_result_qpipe = "pipeline_strategy_result_qpipe.pkl"
    pipeline_strategy_result_qpipe = f'/workspace/qpipe/scripts/baseline_result/{pipeline_strategy_result_qpipe}'
    sharding_strategy = pickle.load(open(pipeline_strategy_result_qpipe, "rb"))

    caliber = lptorch.inner_caliber
    caliber.set_fake() 
    caliber.load_fake_calib_data(f'fake_calib_{model_size}.pkl')
    get_cuda_occupation_by_command()
    model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
    model_pre_and_post = model_pre_and_post.cuda()
    get_cuda_occupation_by_command()
    num_tokens_to_generate = 100
    prompt_length = 512
    bs_token = 16 # how many sentence in a batch
    batched_ids = tokenizer.batch_encode_plus(fetch_prompts(bs_token, prompt_length), padding='max_length', max_length=prompt_length, return_tensors="pt")
    batched_ids = to_device_recursive(dict(batched_ids), 'cuda:0')
    import pdb; pdb.set_trace()
    size = get_iter_variable_size(batched_ids, unit='MB') 
    print(size)
    