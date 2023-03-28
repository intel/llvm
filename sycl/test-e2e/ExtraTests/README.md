> **NOTE**
> This is experimental extension of test suite. Could be dropped or extended
> further at some later point.

# Overview

"ExtraTests" directory contains infrastructure for picking up LIT style tests
from another external directory passed in SYCL_EXTRA_TESTS_SRC.

# CMake parameters

All parameters described in [Readme.md](../README.md#cmake-parameters) are
applicable. Additional parameters for this test category:

***SYCL_EXTRA_TESTS_SRC*** path to directory which contains additional LIT
tests.

# Example

```
# Configure
cmake -G Ninja \
        -DTEST_SUITE_SUBDIRS=SYCL \
        -DTEST_SUITE_LIT=<path/to/llvm-lit> \
        -DCHECK_SYCL_ALL="opencl:cpu,acc,gpu;level_zero:gpu" \
        -DCMAKE_CXX_COMPILER=<path/to/clang++> \
        -DSYCL_EXTRA_TESTS_SRC=<path/to/more/lit/tests/sources>
        ..

# Build and run full list of SYCL tests
ninja ninja check-sycl-all
```
