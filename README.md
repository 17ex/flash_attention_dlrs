# Flash Attention DLRS

## About

This repo contains an implementation of Flash Attention (in development: Flash
Attention v1) in Triton, done for a project for the Deep Learning Research
Kitchen (DLRS) seminar at the University of Tübingen.

## Installation/Setup

## How to run

TODO

## Plans

- finish forward pass, ie.
    - test for correctness
    - benchmark
        - against other implementations, mainly, as of now,
            torch's Attention implementation
        - see if speedup is as expected
        - compare with other Flash Attention implementations
    - implement backward pass
    - write a torch module or function for this
    - implement Flash Attention v2
        - I hope I can finish this before the deadline
    - Future:
        - implement dropout, masking, other functions etc. fused in the kernel.

## Further Resources
