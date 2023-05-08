# Table of contents
 * [Overview](#overview)
 * [Prerequisites](#prerequisites)
 * [Build and run tests](#build-and-run-tests)
 * [CMake parameters](#cmake-parameters)
 * [Special test categories](#special-test-categories)
 * [Creating or modifying tests](#creating-or-modifying-tests)
   * [LIT feature checks](#lit-feature-checks)
   * [llvm-lit parameters](#llvm-lit-parameters)

# Overview
This directory contains SYCL-related tests distributed in subdirectories based
on testing scope.

# Prerequisites

 - DPC++ compiler. Can be built following these
   [instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain)
   or taken prebuilt from [releases](https://github.com/intel/llvm/releases).
 - LIT tools (llvm-lit, llvm-size). They are not available at prebuilts above,
   but can be built in compiler project (e.g. with "ninja check").
 - Target runtime(s) to execute tests on devices. See
   [installation instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#install-low-level-runtime)

# Build and run tests

Get sources

```
git clone https://github.com/intel/llvm
cd llvm/sycl/test-e2e
mkdir build
cd build
```

With compiler tools available in the PATH:

```
# Configure
cmake \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DSYCL_TEST_E2E_TARGETS="opencl:cpu" \
 ..

# Build and Run
make check-sycl-e2e
```

To use ninja build run as:

```
# Configure
cmake -G Ninja ...

# Build and Run
ninja check-sycl-e2e
```

In addition to this, in an in-tree configuration one can enable
`check-sycl-e2e` target for the sycl-toolchain workspace/build by specifying
`SYCL_TEST_E2E_TARGETS` as part of its cmake configuration. For example, like
this:

```
CC=<> CXX=<> python llvm/buildbot/configure.py -o build ... \
  --cmake-opt=-DSYCL_TEST_E2E_TARGETS="level_zero:gpu;opencl:gpu"
  --cmake-opt=-DSYCL_E2E_TESTS_LIT_FLAGS="--param;dump_ir=True"``
```

In an in-tree build, individual tests or groups of tests can be conveniently
run directly from their source paths, using the configured `llvm-lit` script:

```
# Implicitly uses cmake parameters SYCL_BE and SYCL_TARGET_DEVICES, detailed
# below
build/bin/llvm-lit -sv sycl/test-e2e/Basic/aspects.cpp

# Explicitly sets SYCL backend and target device(s), overriding SYCL_BE and
# SYCL_TARGET_DEVICES
build/bin/llvm-lit -sv --param sycl_be=level_zero --param target_devices=cpu,gpu sycl/test-e2e/Basic/aspects.cpp
```

# Cmake parameters

These parameters can be used to configure tests:

***CMAKE_CXX_COMPILER*** - path to DPCPP compiler

***TEST_SUITE_LLVM_SIZE*** - path to llvm-size tool, required for code size
collection

***TEST_SUITE_COLLECT_COMPILE_TIME=OFF*** - can be used to turn off compile
time collection

***TEST_SUITE_COLLECT_CODE_SIZE=OFF*** - can be used to turn off code size
collection

***LLVM_LIT*** - path to llvm-lit tool

***SYCL_TEST_E2E_TARGETS*** - defines selection of multiple SYCL backends with
set of target devices per each to be tested iteratively. Value is
semicolon-separated list of configurations. Each configuration includes backend
separated from comma-separated list of target devices with colon. Example:

```
-DSYCL_TEST_E2E_TARGETS="opencl:cpu;ext_oneapi_level_zero:gpu;ext_oneapi_cuda:gpu;ext_oneapi_hip:gpu;ext_intel_esimd_emulator:gpu"
```

***SYCL_BE*** - SYCL backend to be used for testing. Supported values are:
 - **opencl** - for OpenCL backend;
 - **ext_oneapi_cuda** - for CUDA backend;
 - **ext_oneapi_hip** - for HIP backend;
 - **ext_oneapi_level_zero** - Level Zero backend;
 - **ext_intel_esimd_emulator** - ESIMD emulator backend;


***SYCL_TARGET_DEVICES*** - comma separated list of target devices for testing.
Default value is cpu,gpu,acc. Supported values are:
 - **cpu**  - CPU device available in OpenCL backend only;
 - **gpu**  - GPU device available in OpenCL, Level Zero, CUDA, and HIP backends;
 - **acc**  - FPGA emulator device available in OpenCL backend only;

***OpenCL_LIBRARY*** - path to OpenCL ICD loader library. OpenCL
interoperability tests require OpenCL ICD loader to be linked with. For such
tests OpenCL ICD loader library should be installed in the system or available
at the full path specified by this variable.

***LEVEL_ZERO_INCLUDE*** - path to Level Zero headers.

***LEVEL_ZERO_LIBS_DIR*** - path to Level Zero libraries.

***CUDA_INCLUDE*** - path to CUDA headers.

***CUDA_LIBS_DIR*** - path to CUDA libraries.

***HIP_PLATFORM*** - platform selection for HIP targeted devices.
Defaults to AMD if no value is given. Supported values are:
 - **AMD**    - for HIP to target AMD GPUs
 - **NVIDIA** - for HIP to target NVIDIA GPUs
 
 ***AMD_ARCH*** - flag must be set for when using HIP AMD triple.
 For example it may be set to "gfx906".

***GPU_AOT_TARGET_OPTS*** - defines additional options which are passed to AOT
compilation command line for GPU device. If not specified "-device *" value
is used.

# Special test categories

There are two special directories for extended testing. See documentation at:

 - [ExtraTests](ExtraTests/README.md)
 - [External](External/README.md)

# Creating or modifying tests

## LIT feature checks

Following features can be checked in tests to limit test execution to the
specific environment via REQUIRES, UNSUPPORTED, etc. filters. For example if
REQUIRES:sycl-ls specified, test will run only if sycl-ls tool is available.
If UNSUPPORTED:sycl-ls specified, test will run only if sycl-ls tool is
unavailable.

 * **windows**, **linux** - host OS;
 * **cpu**, **gpu**, **accelerator** - target device;
 * **cuda**, **hip**, **opencl**, **level_zero**, **esimd_emulator** - target
     backend;
 * **sycl-ls** - sycl-ls tool availability;
 * **cm-compiler** - C for Metal compiler availability;
 * **cl_options** - CL command line options recognized (or not) by compiler;
 * **opencl_icd** - OpenCL ICD loader availability;
 * **aot_tool** - Ahead-of-time compilation tools availability;
 * **ocloc**, **opencl-aot** - Specific AOT tool availability;
 * **level_zero_dev_kit** - Level_Zero headers and libraries availability;
 * **cuda_dev_kit** - CUDA SDK headers and libraries availability;
 * **gpu-intel-gen9**  - Intel GPU Gen9  availability;
 * **gpu-intel-gen11** - Intel GPU Gen11 availability;
 * **gpu-intel-gen12** - Intel GPU Gen12 availability;
 * **gpu-intel-dg1** - Intel GPU DG1 availability;
 * **gpu-intel-dg2** - Intel GPU DG2 availability;
 * **gpu-intel-pvc** - Intel GPU PVC availability;
 * **dump_ir**: - compiler can / cannot dump IR;
 * **llvm-spirv** - llvm-spirv tool availability;
 * **llvm-link** - llvm-link tool availability;
 * **fusion**: - Runtime supports kernel fusion;

## llvm-lit parameters

Following options can be passed to llvm-lit tool through --param option to
configure specific single test execution in the command line:

 * **dpcpp_compiler** - full path to dpcpp compiler;
 * **target_devices** - comma-separated list of target devices (cpu, gpu, acc);
 * **sycl_be** - SYCL backend to be used (opencl, ext_oneapi_level_zero,
   ext_oneapi_cuda, ext_oneapi_hip, ext_oneapi_intel_emulator);
 * **dump_ir** - if IR dumping is supported for compiler (True, False);
 * **compatibility_testing** - forces LIT infra to skip the tests compilation
   to support compatibility testing (a SYCL application is built with one
   version of SYCL compiler and then run with different SYCL RT version);
 * **gpu_aot_target_opts** - defines additional options which are passed to AOT
   compilation command line for GPU device. It can be also set by CMake variable
   GPU_AOT_TARGET_OPTS. If not specified "-device *" value is used.
 * **gpu-intel-dg1** - tells LIT infra that Intel GPU DG1 is present in the
   system. It is developer / CI infra responsibility to make sure that the
   device is available in the system. Tests requiring DG1 to run must use proper
   device selector to ensure that. Use SYCL_DEVICE_ALLOWLIST or
   SYCL_DEVICE_FILTER to get proper configuration (see
   [EnvironmentVariables.md](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md));
 * **gpu-intel-dg2** - tells LIT infra that Intel GPU DG2 is present in the
   system. It is developer / CI infra responsibility to make sure that the
   device is available in the system.
 * **gpu-intel-pvc** - tells LIT infra that Intel GPU PVC is present in the
   system. It is developer / CI infra responsibility to make sure that the
   device is available in the system.
 * **extra_environment** - comma-separated list of variables with values to be
   added to test environment. Can be also set by LIT_EXTRA_ENVIRONMENT variable
   in cmake.
 * **level_zero_include** - directory containing Level_Zero native headers,
   can be also set by CMake variable LEVEL_ZERO_INCLUDE.
 * **level_zero_libs_dir** - directory containing Level_Zero native libraries,
   can be also set by CMake variable LEVEL_ZERO_LIBS_DIR.
 * **cuda_include** - directory containing CUDA SDK headers, can be also set by
   CMake variable CUDA_INCLUDE.
 * **cuda_libs_dir** - directory containing CUDA SDK libraries, can be also set
   by CMake variable CUDA_LIBS_DIR.

Example:

```
llvm-lit --param target_devices=gpu --param sycl_be=ext_oneapi_level_zero \
         --param dpcpp_compiler=path/to/clang++ --param dump_ir=True \
         SYCL/External/RSBench
```

