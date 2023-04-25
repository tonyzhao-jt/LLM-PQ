# read partition from file
import pickle
import os 
from qllm.models.OPT.opt import model_cards
from qllm.models.OPT import OPTForCausalLMSeq
import torch 
from transformers import AutoTokenizer
import lptorch
pipeline_strategy_result_qpipe = '/opt/tiger/launch/qsync/QPipe/scripts/part_strategy/qpipe_final_strategy.pkl'
qpipe_partition_strategies = pickle.load(open(pipeline_strategy_result_qpipe, "rb"))
# print it human readable
os.environ['SET_DECODERS_META'] = "1"
model_size = "30b"
config = model_cards[model_size]
loaded_llm_cpu = OPTForCausalLMSeq._from_config(config, torch_dtype=torch.float16)
loaded_llm_cpu.eval() # eval mode
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b")
caliber = lptorch.inner_caliber
caliber.set_fake() 
caliber.load_fake_calib_data(f'fake_calib_{model_size}.pkl')
on_device_0 = loaded_llm_cpu.shard_model(qpipe_partition_strategies, 0)
on_device_0 = on_device_0.cuda()
on_device_0.init_kv_cache(4, 512, 100, request_id=1)
request_token = (torch.rand(4, 512, 7168, dtype=torch.float16).cuda(), None, None, True, 1)
request_token = (torch.rand(4, 1, 7168, dtype=torch.float16).cuda(), None, None, True, 1)
request_token = on_device_0.decode(request_token)
import pdb; pdb.set_trace()
