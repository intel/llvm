# Unified Runtime Benchmark Runner

Scripts for running performance tests on SYCL and Unified Runtime.

## Benchmarks

- [Velocity Bench](https://github.com/oneapi-src/Velocity-Bench)
- [Compute Benchmarks](https://github.com/intel/compute-benchmarks/)

## Running

`$ ./main.py ~/benchmarks_workdir/ ~/llvm/build/`

This will download and build everything in `~/benchmarks_workdir/` using the compiler in `~/llvm/build/`, and then run the benchmarks. The results will be stored in `benchmark_results.md`.

The scripts will try to reuse the files stored in `~/benchmarks_workdir/`, but the benchmarks will be rebuilt every time. To avoid that, use `-no-rebuild` option.

## Requirements

### Python

dataclasses-json==0.6.7

### System

libopencv-dev
