# LLM-PQ
LLM-PQ argues that the assumption of infinite requests in LLM serving is not necessarily valid. We emphasize the importance of efficiently processing workload-similar offline batch processing tasks while also maximizing the utilization of GPUs acquired at different points in time.

## Publications
- (Poster) LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization (PPoPP' 24). 

## Install
```bash
    git clone --recursive https://github.com/tonyzhao-jt/LLM-PQ.git
    python3 pip install -e .
```



### Optimizer
LLM-PQ's optimizer utilize the support from the gurobi. You need to first install gurobi. To use gurobi, put the [web license](https://license.gurobi.com/manager/licenses) under `/opt/gurobi/` or under `configs`:

Elsewise if you run the optimization command, you will get:
```bash
    ERROR:LLM-PQ:Please install gurobi and put the license file under /opt/gurobi/
```

### QLLM
The LLM-PQ's runtime relies on the QLLM / LPTorch. Which is our customized lib for quantized LLM and quantized operators.

### EXTRME CAREFUL
When you use GPU with cap <= 70. please run the update.sh in the 3rd_party of LPTorch to mannually compile and install the bitsandbytes.


### Using dummy for inference
Quantization

