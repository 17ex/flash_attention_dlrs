# Flash Attention DLRS

## About

This repo contains an implementation of Flash Attention (in development: Flash
Attention v1) in Triton, done for a project for the Deep Learning Research
Kitchen (DLRS) seminar at the University of Tübingen.

## Installation/Setup

## How to run

TODO

## Plans

- Forward pass:
    - Change math stuff in inner loop to save some calculations
        - Aka use the proper Flash Attention 2 inner loop
- benchmark
    - against other implementations, mainly, as of now,
        torch's Attention implementation
    - see if speedup is as expected
    - compare with other Flash Attention implementations
- implement backward pass
    - WIP
- write a torch module or function for this
- Future:
    - implement dropout, masking, other functions etc. fused in the kernel.


## Current issues

- Memory requirements etc. incorrect
- Forward pass not proper flash attention 2
- Backward pass is WIP

## Further Resources
