# Table of contents
 * [Overview](#overview)
 * [Prerequisites](#prerequisites)
 * [Developers workflow](#developers-workflow)
 * [Standalone configuration](#standalone)
 * [CMake parameters](#cmake-parameters)
 * [Special test categories](#special-test-categories)
 * [Creating or modifying tests](#creating-or-modifying-tests)
   * [LIT feature checks](#lit-feature-checks)
   * [llvm-lit parameters](#llvm-lit-parameters)
 * [sycl/detail/core.hpp header file](#sycl/detail/core.hpp)

# Overview
This directory contains SYCL-related tests distributed in subdirectories based
on testing scope. They follow the same style as other LIT tests but have a minor
but crucial difference in the behavior of certain directives.

First, some background. SYCL end-to-end tests take much longer to compile
than they spend executing on a device. As such, we want to be able to
structure the test in such a way that we compile it once and then execute
multiple times on different devices (via `ONEAPI_DEVICE_SELECTOR`) to get
required test coverage. The issue here is that the standard approach to `RUN`
directives and substitutions doesn't allow us to expand a line into only
dynamically known number of commands.

To overcome that, we introduce `%{run}` *expansion* that generates multiple
commands from a single `RUN` directive - one per device in `sycl_devices`
parameter (more on the parameter below). There is a small number of tests that
either need multiple devices or target the device selector itself. For such
tests we have a regular `%{run-unfiltered-devices}` substitution that doesn't
set a `ONEAPI_DEVICE_SELECTOR` filter nor does it create multiple commands.
Technically, this change is implemented by creating a custom LIT test
[format](/sycl/test-e2e/format.py) that inherits from `lit.formats.ShTest`.

This custom LIT test format also overrides the meaning of
`REQUIRES`/`UNSUPPORTED`/`XFAIL` directives, although in a natural way that
suits the `%{run}` expansion described above. First, "features" are split into
device independent (e.g. "linux" or "cuda_dev_kit") and device dependent
("cpu/gpu/accelerator", "opencl/cuda/hip/level_zero" and multiple "aspect-\*").
Second, for each device in `sycl_devices` LIT parameter, we check if it satisfies
the conditions in `UNSUPPORTED`/`REQUIRES` rules. If none of the devices do,
the entire test is skipped as `UNSUPPORTED`. Otherwise, if multiple such devices
are supported we do an additional filtering treating any `XFAIL` directives same
way as `UNSUPPORTED`. If only one device is matched by `UNSUPPORTED`/`REQUIRES`
filtering, then `XFAIL` behaves as an actual "expected to fail" marking.

For any device left after the filtering above we expand each `RUN` directive
(including multi-line ones) containing `%{run}` expansion into one command per
device, replacing `%{run}` with

```
env ONEAPI_DEVICE_SELECTOR=<device_matching_requirements> [Optional run_launcher if that is configured]
```

 while at the same time properly handling `%if` conditions,
meaning that the following works as one would expect it to behave:

```
// RUN: %run %t.out %if cpu %{ 1 %} %else %{ 2 %}`. `%{run-unfiltered-devices}
```
is substituted with just `[Optional run_launcher if that is configured]`.

Another little nuance is `%{sycl_triple}` substitution. It is constructed by
concatenating triples for all the devices from `sycl_devices` supported by a
given test. After that there is also a convenient `%{build}` substitution that
is equivalent to `%clangxx -fsycl -fsycl-targets=%{sycl_triple} %s`.

# Prerequisites

 - Target runtime(s) to execute tests on devices. See
   [installation instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#install-low-level-runtime)
 - DPC++ compiler. Can be built following these
   [instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain)
   or taken prebuilt from [releases](https://github.com/intel/llvm/releases).
 - LIT tools (llvm-lit, llvm-size). They are not available at prebuilts above,
   but can be built in compiler project (e.g. with "ninja check").
   
Last two bullets are only needed for a standalone configuration. During a normal
development workflow the tests are integrated into a normal project
build/configuration.

# Developers workflow
Just build the project according to [Getting Started
Guide](/sycl/doc/GetStartedGuide.md) and setup your environment per [Run simple
DPC++ application
Instructions](/sycl/doc/GetStartedGuide.md#run-simple-dpc-application).
Then use

```
# Either absolute or relative path will work.
llvm-lit <repo_path>/sycl/test-e2e
```

to run SYCL End-to-End tests on all devices configured in the system. Use

```
llvm-lit --param sycl_devices="backend0:device0[;backendN:deviceN]*" <repo_path>/sycl/test-e2e
```

to limit execution to particular devices, where `backend` is one of `opencl`,
`hip`, `cuda`, `level_zero`, and `device` is one of `cpu`, `gpu` or `acc`.

To run individual test use the path to it instead of the top level `test-e2e`
directory.

# Standalone configuration

This is supposed to be used for CI/automatic testing and isn't recommended for a
local development setup.

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
-DSYCL_TEST_E2E_TARGETS="opencl:cpu;level_zero:gpu;cuda:gpu;hip:gpu"
```

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
 
***AMD_ARCH*** - flag may be set for when using HIP AMD triple. For example it
may be set to "gfx906". Otherwise must be provided via the ***amd_arch*** LIT
parameter (e.g., ***--param amd_arch=gfx906***) at runtime via the command line
or via the ***LIT_OPTS*** environment variable.

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

### Auto-detected features

The following features are automatically detected by `llvm-lit` by scanning the
environment:

 * **windows**, **linux** - host OS;
 * **cpu**, **gpu**, **accelerator** - target device;
 * **cuda**, **hip**, **opencl**, **level_zero** - target backend;
 * **sycl-ls** - sycl-ls tool availability;
 * **cm-compiler** - C for Metal compiler availability;
 * **cl_options** - CL command line options recognized (or not) by compiler;
 * **opencl_icd** - OpenCL ICD loader availability;
 * **aot_tool** - Ahead-of-time compilation tools availability;
 * **ocloc**, **opencl-aot** - Specific AOT tool availability;
 * **level_zero_dev_kit** - Level_Zero headers and libraries availability;
 * **cuda_dev_kit** - CUDA SDK headers and libraries availability;
 * **dump_ir**: - compiler can / cannot dump IR;
 * **llvm-spirv** - llvm-spirv tool availability;
 * **llvm-link** - llvm-link tool availability;
 * **fusion**: - Runtime supports kernel fusion;
 * **aspect-\<name\>**: - SYCL aspects supported by a device;
 * **arch-\<name\>** - [SYCL architecture](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc) of a device (e.g. `arch-intel_gpu_pvc`, the name matches what you
   can pass into `-fsycl-targets` compiler flag);

### Manually-set features

The following features are only set if you pass an argument to `llvm-lit` (see
section below). All these features are related to HW detection and they should
be considered deprecated, because we have HW auto-detection functionality in
place. No new tests should use these features:

 * **gpu-intel-gen9**  - Intel GPU Gen9  availability;
 * **gpu-intel-gen11** - Intel GPU Gen11 availability;
 * **gpu-intel-gen12** - Intel GPU Gen12 availability;
 * **gpu-intel-dg1** - Intel GPU DG1 availability;
 * **gpu-intel-dg2** - Intel GPU DG2 availability;
 * **gpu-intel-pvc** - Intel GPU PVC availability;
 * **gpu-intel-pvc-vg** - Intel GPU PVC-VG availability;

Note: some of those features describing whole GPU families and auto-detection
of HW does not provide this functionality at the moment. As an improvement, we
could add those features even with auto-detection, because the only alternative
at the moment is to explicitly list every architecture from a family.

## llvm-lit parameters

Following options can be passed to llvm-lit tool through --param option to
configure specific single test execution in the command line:

 * **dpcpp_compiler** - full path to dpcpp compiler;
 * **sycl_devices** - `"backend0:device0[;backendN:deviceN]*"` where `backend`
    is one of `opencl`, `hip`, `cuda`, `level_zero` and `device` is one of
    `cpu`, `gpu` or `acc`.
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
   ONEAPI_DEVICE_SELECTOR to get proper configuration (see
   [EnvironmentVariables.md](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md));
 * **gpu-intel-dg2** - tells LIT infra that Intel GPU DG2 is present in the
   system. It is developer / CI infra responsibility to make sure that the
   device is available in the system.
 * **gpu-intel-pvc** - tells LIT infra that Intel GPU PVC is present in the
   system. It is developer / CI infra responsibility to make sure that the
   device is available in the system.
 * **gpu-intel-pvc-vg** - tells LIT infra that Intel GPU PVC-VG is present in the
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
 * **run_launcher** - part of `%{run*}` expansion/substitution to alter
   execution of the test by, e.g., running it through Valgrind.

Example:

```
llvm-lit --param dpcpp_compiler=path/to/clang++ --param dump_ir=True \
         SYCL/External/RSBench
```

## sycl/detail/core.hpp

While SYCL specification dictates that the only user-visible interface is
`<sycl/sycl.hpp>` header file we found out that as the implementation and
multiple extensions grew, the compile time was getting worse and worse,
negatively affecting our CI turnaround time. As such, we decided to use
finer-grained includes for the end-to-end tests used in this project (under
`sycl/test-e2e/` folder).

At this moment all the tests have been updated to include a limited set of
headers only. However, the work of eliminating unnecessary dependencies between
implementation header files is still in progress and the final set of these
"fine-grained" includes that might be officially documented and suggested for
customers to use isn't determined yet. **Until then, code outside of this project
must keep using `<sycl/sycl.hpp>` provided by the SYCL2020 specification.**
