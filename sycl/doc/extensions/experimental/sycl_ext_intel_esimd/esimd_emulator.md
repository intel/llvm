# ESIMD kernel execution emulation on host.

## Introduction

ESIMD implementation provides a feature to execute ESIMD kernels on the host
CPU without having actual Intel GPU device in the system - this is ESIMD emulator.
It's main purpose is to provide users with a way to conveniently debug ESIMD code
in their favorite debuggers. Performance is not a priority for now and it will like be quite
low. Since the emulator tries to model massively parallel GPU kernel execution on CPU
hardware, some differences in execution profile may happen, and this must be taken
into account when debugging. Redirecting execution to ESIMD emulator is as simple as
setting an environment variable, no program recompilation is needed. When running a
kernel via the emulator, SYCL runtime will see the emulator as normal GPU device - i.e.
`is_gpu()` test will return true for it.

Due to specifics of ESIMD programming model, usual SYCL host device can't execute
ESIMD kernels. For example, it needs some supporting libraries to emulate various kinds
of barriers, GPU execution threads. It would be impractical for host part of a SYCL ESIMD
app to include or link to all the necessary infrastructure components, as it is not needed
in most cases, when there is no ESIMD code or no debugging is wanted. It would also be
inconvenient or even not possible for users to recompile the app with some switch to
execute ESIMD part on CPU. The environment variable plus a separate back-end solve
both problems. 

ESIMD emulator encompasses a the following main components:
1) The ESIMD emulator plugin which is a SYCL runtime back-end similar to OpenCL or
LevelZero.
2) Host implementations of low-level ESIMD intrinsics such as `__esimd_scatter_scaled`.
3) The supporting infrastructure linked dynamically to the plugin - the `libCM` library.

See a specific section below for main ESIMD emulator limitations.

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
are described in [ESIMD CPU Emulation](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-esimd-cpu-emulation)

## Environment variable

For running ESIMD kernels with the ESIMD_EMULATOR backend, the CM_EMU
library requires 'CM_RT_PLATFORM' environment variable set in order to
specify the target platform you want to emulate.

> `$ export CM_RT_PLATFORM=SKL`

List of target platforms supported by CM_EMU is as follows

- SKL
- BXT
- KBL
- ICLLP
- TGLLP
- RKL
- DG1
- ADLP
- ADLS
- ADLN
- DG2
- MTL
- PVC

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
> `$ ONEAPI_DEVICE_SELECTOR=ext_intel_esimd_emulator:gpu ./a.out`

Please note that ESIMD_EMULATOR backend cannot be picked up as default
device automatically. To enable it, `ext_intel_esimd_emulator:gpu` device must
be specified among other devices explicitly in `ONEAPI_DEVICE_SELECTOR` environment
variable. The emulator device effectively replaces any Intel GPU device for SYCL runtime,
so they can't be used simultaneously by a SYCL offload application process. On the other
hand, it is OK to mix the emulator with non-Intel GPU devices or CPU device in
`ONEAPI_DEVICE_SELECTOR`.

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
tested with above command examples due to ESIMD_EMULATOR's limitations
below. And, if 'CM_RT_PLATFORM' is not set, 'skl' is set by default.

## Limitation
- The emulator is available only on Linux for now. Windows support is WIP.
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
