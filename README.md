# LLM-PQ
Official Repo for: LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization
- LLM-PQ argues that the assumption of **infinite requests** in LLM serving is not necessarily valid. 
- LLM-PQ emphasize the importance of efficiently processing workload-similar **predetermined offline batch processing tasks** 
- But also maximizing the utilization of GPUs acquired at different points in time (**Heterogenous GPU Serving**).
- Specially, LLM-PQ is a **workload-centric** and **device-agnostic** serving framework, takes both workload information and device information for strategy derving.

## Before You Proceed
- Due to historical reasons (this repository was initially built between March and June 2023), LLM-PQ's pipeline is built on top of [PipeEdge](https://github.com/usc-isi/PipeEdge). As a result, its performance may be limited compared to the latest pipeline implementations, such as TGI. However, this also ensures a fair comparison with PipeEdge.
- we are planning to replace the backend in the near future. Stay tuned for updates.

## Install
LLM-PQ is implemented in a top-down view, where
- LLM-PQ: Provides the distributed runtime and optimizer for the better serving plan
- QLLM: the customized LLM workload and its quantized version
- LPTorch: the inner most quantization support for the LM, implement different quantization scheme.

Due to the similar reason, later two's performance is not a SOTA. **If this repo / paper is getting popular 🤑🤑🤑, we will consider merging / updates the later two.**

### Docker (Recommended)
You can use the docker file under the dockerfiles. We also provides one pre-built image:

### Manual
```bash
    git clone --recursive https://github.com/tonyzhao-jt/llm_pq.git
    python3 pip install -e .
```
**Careful**: use GPU with cap <= 70 require recompile of bitsandbytes. We done it for u in setup.py, but if not, please run the update.sh in the 3rd_party of LPTorch to mannually compile and install the bitsandbytes.

## Optimizer Setup
llm_pq's optimizer utilize the support from the gurobi. To use gurobi, put the [web license](https://license.gurobi.com/manager/licenses) under `/opt/gurobi/` or under `configs`:

Else you will get:
```bash
    ERROR:llm_pq:Please install gurobi and put the license file under /opt/gurobi/
```

## Reproduce Results in Paper
### Scripts
We provide all the scripts used in the paper under the `scripts/` folder. Currently, we evaluate the performance and accuracy separately by performing the following:
- Evaluating mixed-precision using modified scripts from GPTQ.
- Evaluating performance using the distributed runtime.

Please note that in the current version, we load the model layer by layer and do not require any additional storage for weight saving. However, this loading process might be relatively slow.

### Graphing
We provides all the graphing scripts under the `notebook/` folder.
- For cost model, you need to profile `gtruth` for prediction error est.



## TODOs if Heated 
1. Faster Loading
We are going to add scripts to distributed runtime and quantization part to make it can be fast deployed in runtime. Stay-tuned.
2. Better Pipeline
Replace PipeEdge's 
3. Simplify model structure
The existing model structure is directly adopted from old transformer lib, introducing many unnecessary ops which could be reduced.

## Citation
If you use LLM-PQ for your research, please cite our [paper](https://dl.acm.org/doi/10.1145/3627535.3638480):
```bibtex
@inproceedings{10.1145/3627535.3638480,
author = {Zhao, Juntao and Wan, Borui and Wu, Chuan and Peng, Yanghua and Lin, Haibin},
title = {POSTER: LLM-PQ:Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization},
year = {2024},
isbn = {9798400704352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627535.3638480},
doi = {10.1145/3627535.3638480},
pages = {460–462},
keywords = {LM serving, heterogenous cluster, quantization},
series = {PPoPP '24}
}
```
