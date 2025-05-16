# Table of contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Developers workflow](#developers-workflow)
* [Standalone configuration](#standalone-configuration)
* [CMake parameters](#cmake-parameters)
* [Special test categories](#special-test-categories)
* [Creating or modifying tests](#creating-or-modifying-tests)
  * [LIT feature checks](#lit-feature-checks)
  * [llvm-lit parameters](#llvm-lit-parameters)
  * [Marking tests as expected to fail](#marking-tests-as-expected-to-fail)
  * [Marking tests as unsupported](#marking-tests-as-unsupported)
* [SYCL core header file](#sycl-core-header-file)
* [Compiling and executing tests on separate systems](#compiling-and-executing-tests-on-separate-systems)
  * [Run only mode](#run-only-mode)
  * [Build only mode](#build-only-mode)
  * [Common Issues with separate build and run](#common-issues-with-separate-build-and-run)

## Overview

This directory contains SYCL-related tests distributed in subdirectories based
on testing scope. They follow the same style as other LIT tests but have a minor
but crucial difference in the behavior of certain directives.

First, some background. SYCL end-to-end tests take much longer to compile than
they spend executing on a device. As such, we want to be able to structure the
test in such a way that we compile it once and then execute multiple times on
different devices (via `ONEAPI_DEVICE_SELECTOR`) to get required test coverage.
The issue here is that the standard approach to `RUN` directives and
substitutions doesn't allow us to expand a line into only dynamically known
number of commands.

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
Second, for each device in `sycl_devices` LIT parameter, we check if it
satisfies the conditions in `UNSUPPORTED`/`REQUIRES` rules. If none of the
devices do, the entire test is skipped as `UNSUPPORTED`. Otherwise, if multiple
such devices are supported we do an additional filtering treating any `XFAIL`
directives same way as `UNSUPPORTED`. If only one device is matched by
`UNSUPPORTED`/`REQUIRES` filtering, then `XFAIL` behaves as an actual "expected
to fail" marking.

For any device left after the filtering above we expand each `RUN` directive
(including multi-line ones) containing `%{run}` expansion into one command per
device, replacing `%{run}` with

```bash
env ONEAPI_DEVICE_SELECTOR=<device_matching_requirements> [Optional run_launcher if that is configured]
```

while at the same time properly handling `%if` conditions, meaning that the
following works as one would expect it to behave:

```c++
// RUN: %run %t.out %if cpu %{ 1 %} %else %{ 2 %}`. `%{run-unfiltered-devices}
```

is substituted with just `[Optional run_launcher if that is configured]`.

Another little nuance is `%{sycl_triple}` substitution. It is constructed by
concatenating triples for all the devices from `sycl_devices` supported by a
given test. After that there is also a convenient `%{build}` substitution that
is equivalent to `%clangxx -fsycl %{sycl_target_opts} %s`.

## Prerequisites

* Target runtime(s) to execute tests on devices. See
  [installation instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#install-low-level-runtime)
* DPC++ compiler. Can be built following these
  [instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain)
  or taken prebuilt from [releases](https://github.com/intel/llvm/releases).
* LIT tools (llvm-lit, llvm-size). They are not available at prebuilts above,
  but can be built in compiler project (e.g. with "ninja check").

Last two bullets are only needed for a standalone configuration. During a normal
development workflow the tests are integrated into a normal project
build/configuration.

## Developers workflow

Just build the project according to
[Getting Started Guide](/sycl/doc/GetStartedGuide.md) and setup your environment
per [Run simple DPC++ application Instructions](/sycl/doc/GetStartedGuide.md#run-simple-dpc-application).

Then use

```bash
# Either absolute or relative path will work.
llvm-lit <repo_path>/sycl/test-e2e
```

to run SYCL End-to-End tests on all devices configured in the system. Use

```bash
llvm-lit --param sycl_devices="backend0:device0[;backendN:deviceN]*" <repo_path>/sycl/test-e2e
```

to limit execution to particular devices, where `backend` is one of `opencl`,
`hip`, `cuda`, `level_zero`, and `device` is one of `cpu`, `gpu` or `acc`.

To run individual test use the path to it instead of the top level `test-e2e`
directory.

## Standalone configuration

This is supposed to be used for CI/automatic testing and isn't recommended for a
local development setup.

Get sources

```bash
git clone https://github.com/intel/llvm
cd llvm/sycl/test-e2e
mkdir build
cd build
```

With compiler tools available in the PATH:

```bash
# Configure
cmake \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DSYCL_TEST_E2E_TARGETS="opencl:cpu" \
 ..

# Build and Run
make check-sycl-e2e
```

To use ninja build run as:

```bash
# Configure
cmake -G Ninja ...

# Build and Run
ninja check-sycl-e2e
```

## CMake parameters

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

```bash
-DSYCL_TEST_E2E_TARGETS="opencl:cpu;level_zero:gpu;cuda:gpu;hip:gpu"
```

In addition, device architecture as shown in sycl-ls is accepted with the
"arch-" prefix. Example:

```bash
-DSYCL_TEST_E2E_TARGETS="cuda:arch-nvidia_gpu_sm_61;level_zero:arch-intel_gpu_bmg_b21"
```

***OpenCL_LIBRARY*** - path to OpenCL ICD loader library. OpenCL
interoperability tests require OpenCL ICD loader to be linked with. For such
tests OpenCL ICD loader library should be installed in the system or available
at the full path specified by this variable.

***LEVEL_ZERO_INCLUDE*** - path to Level Zero headers.

***LEVEL_ZERO_LIBS_DIR*** - path to Level Zero libraries.

***CUDA_INCLUDE*** - path to CUDA headers (autodetected).

***CUDA_LIBS_DIR*** - path to CUDA libraries (autodetected).

***HIP_INCLUDE*** - path to HIP headers (autodetected).

***HIP_LIBS_DIR*** - path to HIP libraries (autodetected).

***AMD_ARCH*** - flag may be set for when using HIP AMD triple. For example it
may be set to "gfx906". Otherwise must be provided via the ***amd_arch*** LIT
parameter (e.g., ***--param amd_arch=gfx906***) at runtime via the command line
or via the ***LIT_OPTS*** environment variable.

***GPU_AOT_TARGET_OPTS*** - defines additional options which are passed to AOT
compilation command line for GPU device. If not specified "-device *" value is
used.

***USE_DEBUG_LIBRARIES*** - enable the use of Windows debug libraries by setting
this to "ON". "OFF" by default.

## Special test categories

There are two special directories for extended testing. See documentation at:

* [ExtraTests](ExtraTests/README.md)
* [External](External/README.md)

## Creating or modifying tests

### LIT feature checks

Following features can be checked in tests to limit test execution to the
specific environment via REQUIRES, UNSUPPORTED, etc. filters. For example if
REQUIRES:sycl-ls specified, test will run only if sycl-ls tool is available. If
UNSUPPORTED:sycl-ls specified, test will run only if sycl-ls tool is
unavailable.

#### Auto-detected features

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
* **aspect-\<name\>**: - SYCL aspects supported by a device;
* **arch-\<name\>** - [SYCL architecture](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc)
  of a device (e.g. `arch-intel_gpu_pvc`, the name matches what you can pass
  into `-fsycl-targets` compiler flag);
* **gpu-intel-dg2**  - Intel GPU DG2 availability; Automatically set if device
  architecture belongs to DG2 family;
* **gpu-intel-gen11**  - Intel GPU Gen11 availability; Automatically set if device
  architecture belongs to Gen11 family;
* **gpu-intel-gen12**  - Intel GPU Gen12 availability; Automatically set if device
  architecture belongs to Gen12 family;

#### Manually-set features

The following features are only set if you pass an argument to `llvm-lit` (see
section below). All these features are related to HW detection and they should
be considered deprecated, because we have HW auto-detection functionality in
place. No new tests should use these features:

* **gpu-intel-pvc** - Intel GPU PVC availability;
* **gpu-intel-pvc-vg** - Intel GPU PVC-VG availability;

### Use the LLVM SPIR-V Backend to generate SPIR-V code
It's possible to use the LLVM SPIR-V Backend instead of the `llvm-spirv` tool
to convert LLVM IR to SPIR-V. This feature must be set manually by passing the
`spirv-backend` parameter to `llvm-lit`.

### llvm-lit parameters

Following options can be passed to llvm-lit tool through --param option to
configure specific single test execution in the command line:

* **dpcpp_compiler** - full path to dpcpp compiler;
* **sycl_devices** - `"backend0:device0[;backendN:deviceN]*"` where `backend` is
  one of `opencl`, `hip`, `cuda`, `level_zero` and `device` is one of `cpu`,
  `gpu` or `acc`. Device may also be device architecture as listed in
  `sycl-ls --verbose` prefixed with `arch-`. Example:
  `level_zero:arch-intel_gpu_bmg_g21`
* **dump_ir** - if IR dumping is supported for compiler (True, False);
* **compatibility_testing** - forces LIT infra to skip the tests compilation to
  support compatibility testing (a SYCL application is built with one version of
  SYCL compiler and then run with different SYCL RT version);
* **gpu_aot_target_opts** - defines additional options which are passed to AOT
  compilation command line for GPU device. It can be also set by CMake variable
  GPU_AOT_TARGET_OPTS. If not specified "-device *" value is used.
* **gpu-intel-dg2** - tells LIT infra that Intel GPU DG2 is present in the
  system. It is developer / CI infra responsibility to make sure that the device
  is available in the system.
* **gpu-intel-pvc** - tells LIT infra that Intel GPU PVC is present in the
  system. It is developer / CI infra responsibility to make sure that the device
  is available in the system.
* **gpu-intel-pvc-vg** - tells LIT infra that Intel GPU PVC-VG is present in the
  system. It is developer / CI infra responsibility to make sure that the device
  is available in the system.
* **extra_environment** - comma-separated list of variables with values to be
  added to test environment. Can be also set by LIT_EXTRA_ENVIRONMENT variable
  in CMake.
* **extra_system_environment** - comma-separated list of variables to be
  propagated from the host environment to test environment. Can be also set by
  LIT_EXTRA_SYSTEM_ENVIRONMENT variable in CMake.
* **level_zero_include** - directory containing Level_Zero native headers, can
  be also set by CMake variable LEVEL_ZERO_INCLUDE.
* **level_zero_libs_dir** - directory containing Level_Zero native libraries,
  can be also set by CMake variable LEVEL_ZERO_LIBS_DIR.
* **cuda_include** - directory containing CUDA SDK headers, can be also set by
  CMake variable CUDA_INCLUDE (autodetected).
* **cuda_libs_dir** - directory containing CUDA SDK libraries, can be also set
  by CMake variable CUDA_LIBS_DIR (autodetected).
* **hip_include** - directory containing HIP SDK headers, can be also set by
  CMake variable HIP_INCLUDE (autodetected).
* **hip_libs_dir** - directory containing HIP SDK libraries, can be also set
  by CMake variable HIP_LIBS_DIR (autodetected).
* **run_launcher** - part of `%{run*}` expansion/substitution to alter execution
  of the test by, e.g., running it through Valgrind.

Example:

```bash
llvm-lit --param dpcpp_compiler=path/to/clang++ --param dump_ir=True \
         SYCL/External/RSBench
```

### Marking tests as expected to fail

Every test should be written in a way that it is either passed, or it is skipped
(in case it is not compatible with an environment it was launched in).

If for any reason you find yourself in need to temporary mark test as expected
to fail under certain conditions, you need to submit an issue to the repo to
analyze that failure and make test passed or skipped.

Once the issue is created, you can update the test by adding `XFAIL` and
`XFAIL-TRACKER` directive:

```c++
// GPU driver update caused failure
// XFAIL: level_zero
// XFAIL-TRACKER: PRJ-5324

// Sporadically fails on CPU:
// XFAIL: cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/DDDDD
```

If you add `XFAIL` without `XFAIL-TRACKER` directive,
`no-xfail-without-tracker.cpp` test will fail, notifying you about that.

### Marking tests as unsupported

Some tests may be considered unsupported, e.g.:

* the test checks the feature that is not supported by some backend / device /
  OS / etc.
* the test is flaky or hangs, so it can't be marked with `XFAIL`.

In these cases the test can be marked with `UNSUPPORTED`. This mark should be
followed by either `UNSUPPORTED-INTENDED` or `UNSUPPORTED-TRACKER` depending on
whether the test is not intended to be run with some feature at all or it was
temporarily disabled due to some issue.

```c++
// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: only supported by backends with SPIR-V IR

// Sporadically fails on DG2.
// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/DDDDD
// *OR*
// UNSUPPORTED-TRACKER: PRJ-1234
```

If you add `UNSUPPORTED` without `UNSUPPORTED-TRACKER` or `UNSUPPORTED-INTENDED`
directive, the `no-unsupported-without-tracker.cpp` test will fail, notifying
you about that.

To disable the test completely, you can use:

```c++
// USNUPPORTED: true
```

Note: please avoid using `REQUIRES: TEMPORARY_DISABLED` for this purpose, it's a
non-standard mechanism. Use `UNSUPPORTED: true` instead, we track `UNSUPPORTED`
tests using the mechanism described above. Otherwise the test risks remaining
untraceable.

### SYCL core header file

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
customers to use isn't determined yet. **Until then, code outside of this
project must keep using `<sycl/sycl.hpp>` provided by the SYCL2020
specification.**

### Compiling and executing tests on separate systems

The execution of e2e tests can be separated into compilation and execution
stages via the `test-mode` lit parameter. This allows us to reduce testing time
by compiling tests on more powerful systems and reusing the binaries on other
machines. By default the `test-mode` parameter is set to `full`, indicating that
both stages will run. This parameter can be set to `build-only`, or `run-only`,
to only run the compilation stage, or the execution stage respectively.

#### Run only mode
  
  Pass: `--param test-mode=run-only`

  In this mode, tests will not be compiled, they will only run. To do this only
  the `RUN:` lines that contain a "run" expansion will be executed (`%{run}`,
  `%{run-unfiltered-devices}`, or `%{run-aux}`). Since tests are not compiled in
  this mode, for any test to pass the test binaries should already be in the
  `test_exec_root` directory, either by having ran `full` or `build-only` modes
  previously on the system, or having transferred the test binaries into that
  directory. To mark a test as expected to fail at run-time the `XFAIL`
  expression should use runtime features, such as `run-mode` or device-specific
  features.

  `%{run-aux}` is an empty expansion and executes a line as is, without
  expanding for each selected device and without using the `run_launcher`.

#### Build only mode

  Pass: `--param test-mode=build-only`

  This mode can be used to compile all test binaries that can be built on the
  system. To do this `REQUIRES`, and `UNSUPPORTED` statements are handled
  differently to accommodate for the fact that in `build-only` mode we do not
  have any devices, and as a result no device-specific features. Instead of
  considering these features as missing, we assign a third "unknown" value to 
  them. When evaluating an expression it will result in an unknown value if its 
  result could be changed by setting the unknown features to either true or 
  false. i.e., `false || unknown = unknown` but `true || unknown = true`. If an 
  expression's final value is unknown we consider it to have met the
  requirements. The list of device-agnostic features that are not considered
  unknown in `build-only` is found in the `E2EExpr.py`.

  The triples to compile in this mode are set via the `sycl_build_targets` lit
  parameter. Valid build targets are: `spir`,`nvidia`, `amd`, `native_cpu`.
  These correspond to `spir64`, `nvptx64-nvidia-cuda`, `amdgcn-amd-amdhsa`, and
  `native_cpu` triples respectively. Each build target should be separated with
  a semicolon. This parameter is set to `all` by default, which enables
  autodetection for the available build targets. A test can be marked as
  requiring, or not supporting a particular triple via the `target-*` features.
  Build targets are selected if they are able to pass the test's requirements
  independent of the availability of other build targets. This is done to avoid
  having to deal with a boolean satisfiability problem. For example,
  `REQUIRES: target-spir && target-nvidia` will always be marked as unsupported
  since it requires multiple targets simultaneously. Instead we can use
  `any-target-is-*` features in this case, to check if a target is available in
  the current lit configuration.

  When executing the test in `build-only`, all `RUN:` lines that do not have a
  run expansion will execute.

  The `build-mode` feature is added when in this mode.

  Some examples of `REQUIRES`/`UNSUPPORTED` in build-only:
  If `linux` and `zstd` are available, and `sycl_build_targets` is set to
  `spir;amd`
  * `REQUIRES: linux && zstd`: This would be supported, this is treated normally
  since both features are device-agnostic.
  * `REQUIRES: linux && sg-32`: Despite the `sg-32` feature not being available,
  this would be supported. Since the `sg-32` is a device-specific feature it is
  evaluated as unknown in this expression.
  * `REQUIRES: windows && sg-32`: This would be unsupported. `sg-32` would be
  evaluated as unknown, and `windows` would evaluate as false. The fact that we
  have an unknown value does not affect the end result, since the result of an
  `&&` expression where one sub-expression is false is always false.
  * `REQUIRES: windows || sg-32`: this would be supported. Here because the
  result of the `||` expression would change if we considered `sg-32` to be
  either true or false the overall expression evaluates to unknown.
  * `UNSUPPORTED: !sg-32`: this would be supported. `sg-32` is evaluated as
  unknown, and the negation of unknown is also unknown.
  * `REQUIRES: target-spir`: This will be supported, and only the `spir64`
  triple will be selected.
  * `REQUIRES: target-spir && target-amd`: This will not be supported. When
  checking if `target-spir` should be selected, `target-amd` will not be an
  available feature, and vice versa when checking `target-amd`.
  * `REQUIRES: target-spir && any-target-is-amd`: This will be supported, but
  only the `spir64` triple will be selected.

#### Common Issues with separate build and run

A number of extra considerations need to be taken to write tests that are able
to be compiled and executed on separate machines.

* Tests that build and execute multiple binaries need to be written such that
the output of each compilation has a different name. This way no files are
overwritten, and all the necessary binaries can be transferred to the running
system. For example, instead of setting the output of all compilation steps to
a file named `%t.out`, we can number them `%t1.out`, `%t2.out`, and so on.

* Two scenarios need to be considered for tests that expectedly fail:
  * Tests that are expected to fail on compilation, and thus also during
  execution, need to be marked as `XFAIL` for a device-agnostic feature, or
  with `XFAIL: *`. This is due to the fact that there are no devices in
  `build-only` mode. For example if a test cannot compile for a triple, then it
  should be marked as `XFAIL` for the corresponding build target feature, rather
  than a backend feature.
  * If the expected failure occurs during run-time we will need to mark the test
  with `XFAIL` with an expression dependent on runtime features. If it is
  expected to fail for any device at run-time we can use `XFAIL: run-mode`,
  This must be done because otherwise the test would compile and pass on
  `build-only` mode and be reported as an `XPASS`.

* To separate compilation and execution of tests, `RUN:` lines are filtered in
`build-only` and `run-only` mode based on the presence of "run" expansions.
  * Any line that is meant to execute the test binary should be marked with
  `%{run}` or `%{run-unfiltered-devices}` so that it is not ran in `build-only`,
  and the `run_launcher` substitution is properly employed.
  * The `%{run-aux}` expansion can be used if a `RUN:` line that does not
  execute a test binary needs to be ran in `run-only`.

* CPU AOT compilation will target the ISA of the host CPU, thus compiling
these tests on a different system will lead to failures if the build system and
run system support different ISAs. To accommodate this, these compilations
should be delayed to the "run" stage by using the `%{run-aux}` markup.
