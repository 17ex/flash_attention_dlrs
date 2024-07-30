# Flash Attention DLRS

## About

This repo contains an implementation of Flash Attention v2 in Triton,
done for a project for the Deep Learning Research
Kitchen (DLRS) seminar at the University of Tübingen.

## Installation/Setup

- Set up and activate a python venv:
    - ```python -m venv your-venv-name```
    - ```source your-venv-name/bin/activate```
- Install dependencies: ```pip install -r requirements.txt```

## How to run

- These files are runnable:
    - ```bench.py```: Benchmark my implementation vs others
    - ```plot_bench_results.py```: Plot (and run, if no data exists yet)
        benchmark results from ```bench.py```
    - ```test_correctness.py```: Test my implementation for correctness
        (FP32 comparison with torch)
    - ```test_torch.py```: Test my implementation with torch autograd

- You can set up parameters within the files and run them.
- ```flash_attention_torch.py``` contains torch abstractions for my Flash
    Attention implementation. You can import them and use them in torch.
- If you plan to run/use anything here, please consider commenting out a lot of
    configs in ```autotune_configs.py```. If you don't, especially in
    benchmarking, autotuning will take very long.

## Plans

- implement deterministic backward pass
    - WIP
- Future:
    - implement dropout, masking, other functions etc. fused in the kernel.


## Current issues

- (Probably?) Only CUDA devices supported, memory requirements based off NVIDIA GA102 GPUs
    - I don't have other devices to develop/check for
- Backwards pass (not the deterministic version)
    - A bit fishy: If run the first time, results can be completely wrong,
        but after that, it works reliably.
        Need to check out what is going on there.
- Backwards pass (deterministic version)
    - DOES NOT WORK AT ALL!
    - Do not use it
    - It's also not deterministic
    - Very much WIP
- Autotuning may lead to out of memory errors. If that's the case,
    then increase ```SAFETY_MARGIN_MEM_FACTOR``` in ```autotune_configs.py```
    and try again.

## Further Resources

TODO Add helpful links, papers etc.
