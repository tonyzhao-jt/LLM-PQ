# SplitQuant / LLM-PQ: Resource-Efficient LLM Offline Serving on Heterogeneous GPUs via Phase-Aware Model Partition and Adaptive Quantization

This is the official repository for **SplitQuant** (IEEE Cluster 2025) and its preliminary work, **LLM-PQ** (ACM PPoPP'24 Poster).

<p align="center">
 | <a href="https://arxiv.org/abs/2403.01136"><b>Full Paper</b></a>｜  <a href="https://dl.acm.org/doi/10.1145/3627535.3638480"><b>Poster</b></a> |
</p>

-----

### Backends

This project provides two backends for reproducing results and for practical deployment:

1.  **This Repository (Research Prototype Backend)**: This backend was built on PipeEdge and is used to reproduce the specific results and baselines presented in the paper. It serves as the primary research artifact.
2.  **High-Performance vLLM Backend**: For users seeking higher performance and easier deployment, a more optimized backend based on vLLM is available at: **[https://github.com/tonyzhao-jt/LLMPQ-vLLM](https://github.com/tonyzhao-jt/LLMPQ-vLLM)**

-----

SplitQuant is a serving framework designed for offline batch processing of Large Language Model (LLM) workloads on heterogeneous GPU clusters. It introduces a workload-centric and device-agnostic serving strategy, incorporating phase-aware model partitioning and adaptive quantization to optimize resource utilization.

## Installation

### Docker (Recommended)

Pre-built Docker images are available for different GPU architectures:

```bash
# For GPUs with compute capability <= 7.0 (e.g., V100), which require a custom build of bitsandbytes
docker pull springtonyzhao/llmpq:v100

# For newer GPUs (e.g., A100)
docker pull springtonyzhao/llmpq:a100
```

### Manual Installation

```bash
git clone --recursive https://github.com/tonyzhao-jt/llm_pq.git
cd llm_pq
pip3 install -e .
```

**Note**: GPUs with compute capability 7.0 or lower require a manual compilation of `bitsandbytes`. The `setup.py` script attempts to handle this. If it fails, please run the `update.sh` script in `3rd_party/LPTorch` to manually compile and install it.

#### Common Errors

  - `BuilderConfig 'allenai--c4' not found. Available: ...`: Please change the data loading script in the GPTQ component to:
    ```python
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    ```
  - `ERROR: Could not install packages due to an OSError:`: This can sometimes occur during `pip install -e .`. Simply running the command again usually resolves the issue.

## Optimizer Setup

The optimizer relies on Gurobi. Please obtain a [web license](https://license.gurobi.com/manager/licenses) and place the `gurobi.lic` file either in `/opt/gurobi/` or in the `configs/` directory of this project. Without a valid license, you will encounter the following error:

```bash
ERROR:llm_pq: Please install gurobi and put the license file under /opt/gurobi/
```

## Reproduce Results in Paper

### Scripts

All scripts used for the paper's experiments are located in the `scripts/` directory. The evaluation is split into two parts:

  - **Accuracy**: Mixed-precision accuracy is evaluated using modified scripts from GPTQ.
  - **Performance**: End-to-end serving performance is measured using the distributed runtime.

In the current implementation, models are loaded layer by layer at runtime, which avoids the need for saving quantized weights to disk but can make the initial loading process slower.

### Graphing

All scripts for generating plots and figures from the paper are available in the `notebook/` directory. For cost model validation, you will need to profile the `gtruth` to estimate prediction errors.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@misc{zhao2024llmpqservingllmheterogeneous,
      title={LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization}, 
      author={Juntao Zhao and Borui Wan and Yanghua Peng and Haibin Lin and Chuan Wu},
      year={2024},
      eprint={2403.01136},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.01136}, 
}
```