# ESIMD kernel execution emulation on host.

## Introduction

ESIMD_EMULATOR is emulator mode for running ESIMD kernels. Under Linux
environment, programmers can run ESIMD kernels on CPU without actual
Intel GPU hardware using ESIMD_EMULATOR backend. This backend is a
virtual GPU plug-in device that runs ESIMD kernels as if they are
running on GPU hardware. The backend emulates GPU hardware using
software multi-threading supported by Linux host machine. This means
that the executable file can be run and debugged like normal x86_64
Linux application.

## Requirements

ESIMD_EMULATOR backend uses [CM_EMU
library](https://github.com/intel/cm-cpu-emulation) for emulating GPU
using software multi-threading. The library can be either provided as
separate pre-installed library in host machine or built as part of
open-source Intel DPC++ compiler. Required version for CM_EMU is
[1.0.20](https://github.com/intel/cm-cpu-emulation/releases/tag/v2022-02-11)
or later. In order to have CM_EMU library as part of Intel DPC++
compiler for ESIMD_EMULATOR backend, the library needs to be built
during ESIMD_EMULATOR plug-in software module generation. Details on
building CM_EMU library for ESIMD_EMULATOR such as required packages
are descirbed in [ESIMD CPU Emulation](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-esimd-cpu-emulation)

## Command line option / environment variable options

There is no special command line option or environment variable
required for building and running ESIMD kernels with ESIMD_EMULATOR
backend.

## Running ESIMD code under emulation mode

Compilation step for ESIMD kernels prepared for ESIMD_EMULATOR backend
is same as for OpenCL and Level Zero backends. Full runnable code
sample used below can be found on the [github
repo](https://github.com/intel/llvm-test-suite/blob/intel/SYCL/ESIMD/vadd_usm.cpp).

To compile using the open-source Intel DPC++ compiler:
> `$ clang++ -fsycl vadd_usm.cpp`

To compile using Intel(R) OneAPI Toolkit:
> `$ dpcpp vadd_usm.cpp`

To run under emulation through ESIMD_EMULATOR backend:
> `$ SYCL_DEVICE_FILTER=ext_intel_esimd_emulator:gpu ./a.out`

## Running ESIMD examples from [ESIMD test suite](https://github.com/intel/llvm-test-suite/tree/intel/SYCL/ESIMD) on github with ESIMD_EMULATOR backend

```
# Get sources
git clone https://github.com/intel/llvm-test-suite
cd llvm-test-suite
mkdir build && cd build

# Configure for make utility with compiler tools available in $PATH
cmake \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DTEST_SUITE_SUBDIRS=SYCL \
 -DSYCL_BE="ext_intel_esimd_emulator" \
 -DSYCL_TARGET_DEVICES="gpu" \
 ..

# Build and Run
make check

# Or, for Ninja utility
cmake -G Ninja \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DTEST_SUITE_SUBDIRS=SYCL \
 -DSYCL_BE="ext_intel_esimd_emulator" \
 -DSYCL_TARGET_DEVICES="gpu" \
 ..

# Build and Run
ninja check

```

Note that only [ESIMD Kernels](https://github.com/intel/llvm-test-suite/tree/intel/SYCL/ESIMD) are
tested with above command examples due to ESIMD_EMULATOR's limiations
below.

## Limitation

- ESIMD_EMULATOR has limitation on number of threads under Linux. As
software multi-threading is used for emulating hardware threads,
number of threads being launched for kernel execution is limited by
the max number of threads supported by Linux host machine.

- ESIMD_EMULATOR supports only ESIMD kernels. This means kernels
written for SYCL cannot run with ESIMD_EMULATOR backend. This also
means that kernels containing both SYCL and ESIMD code cannot run with
ESIMD_EMULATOR, unlike GPU backends like OpenCL or Level Zero.

- ESIMD_EMULATOR cannot run in parallel with Host Device.

## TODO

- Windows environment support

- Support for arithmetic operations for 16-bit half floating point
number type
