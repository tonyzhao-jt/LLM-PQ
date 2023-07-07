# Sample
# We provide some single-card and single-node results here for validation

# Usage
Run gen fake calib
```bash
    cd fake_calib && bash gen_fake.sh && cd ..
```
## Perf Mode
If you are only concerned with performance rather than accuracy, or if you simply want to benchmark the performance in some cases, we recommend using the following commands:

```bash
os.environ['SET_DECODERS_META'] = "1"
os.environ['PERF_MODE'] = "0"
```
1. First option create empty decoder[None] for each device then load weight accordingly.
2. The second option tells QPipe to randomly initialize the weights. By doing so, you can greatly reduce the CPU usage when loading the model and quickly obtain benchmark results.
3. However, in perf model, the performance may not consistent to the weight-loadded case. 

Enable perf-mode by passing
`python xx.py --perf-mode`

# Sample
## UU Sample
We provide sample run for uniform-q + uniform partition
```
    bash sample_run.sh
```
## 

