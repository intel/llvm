# Table of contents
 * [Overview](#overview)
 * [Execution](#execution)
 * [Main parameters](#main-parameters)
 * [LIT parameters accepted by LIT executor](#lit-parameters-accepted-by-lit-executor)
 * [LIT features which can be used to configure test execution](#lit-features-which-can-be-used-to-configure-test-execution)

# Overview
This directory contains SYCL-related tests distributed in subdirectories:
 - Basic - tests used for sanity testing. Building, executing and checks are
   defined using insource comments with LIT syntax.
 - External - contains infrastructure for running tests which sources are
   stored outside of this repository.

# Execution
```
git clone <GIT_REPO> # e.g. https://github.com/intel/llvm-test-suite
cd llvm-test-suite
mkdir build
cd build
# configuring test execution (selecting compiler version, target BE and target device)
cmake -G Ninja \
        -DTEST_SUITE_SUBDIRS=SYCL \
        -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> \
        -DCHECK_SYCL_ALL="PI_OPENCL:cpu,acc,gpu,host;PI_LEVEL_ZERO:gpu,host" \
        -C<CMAKE_CHASHED_CONFIG> \
        -DSYCL_EXTERNAL_TESTS="RSBench" \
        ..
# Building full list of tests in subdir
ninja check
# or
llvm-lit .
# Get list of available tests
llvm-lit . --show-tests
# Run specific test
llvm-lit <path_to_test>
# Run tests with parameters
llvm-lit --param target_devices=host,gpu --param sycl_be=PI_LEVEL_ZERO \
        --param dpcpp_compiler=path/to/clang++ --param dump_ir=True .
```

Notes:
 - it is assumed that LIT framework, FileCheck and other LIT dependencies are
available in the same directory with llvm-lit.
 - compiler variant as well as compile/link options are defined in cashed cmake
 configurations:
   - [dpcpp.cmake](../../cmake/caches/dpcpp.cmake)
   - [clang_fsycl.cmake](../../cmake/cashes/clang_fsycl.cmake)
   - [clang_fsycl_cuda.cmake](../../cmake/cashes/clang_fsycl_cuda.cmake)
 - compiler is taken from environment.

# Main parameters
It is possible to change test scope by specifying test directory/file in first
argument to for the lit-runner.py script.

***CMAKE_CXX_COMPILER*** should point to the DPCPP compiler

***SYCL_TARGET_DEVICES*** defines comma separated target device types (default
value is cpu,gpu,acc,host). Supported target_devices values are:
 - **cpu**  - CPU device available in OpenCL backend only;
 - **gpu**  - GPU device available in OpenCL, Level Zero and CUDA backends;
 - **acc**  - FPGA emulator device available in OpenCL backend only;
 - **host** - SYCL Host device availabel with all backends.

***SYCL_BE*** defined SYCL backend to be used for testing (default is
PI_OPENCL).
Supported sycl_be values:
 - PI_OPENCL - for OpenCL backend;
 - PI_CUDA - for CUDA backend;
 - PI_LEVEL_ZERO - Level Zero backend.

***CHECK_SYCL_ALL*** allows selection of multiple SYCL backends with set of
target devices per each to be tested iteratively. Value may contain semicolon-
separated list of configurations. Each configuration includes backend separated
from comma-separated list of target devices with colon (e.g.
-DCHECK_SYCL_ALL="PI_OPENCL:cpu,host;PI_LEVEL_ZERO:gpu,host"). The testing is
done using check-sycl-all target. It is recommended to pass -k0 parameter to
build command line to avoid break execution on test failures for the first
backend.

***SYCL_EXTERNAL_TESTS*** semicolon-separate names of external SYCL applications
which are built and run as part of the testing. Name is equal to subdirectory in
[External](External) containing driver for building and running the application
in llvm-test-suite infrastructure (e.g. -DSYCL_EXTERNAL_TESTS=RSBench). Source
code of external application can be downloaded from external repo as part of the
build or provided in CMake variable <APPNAME>_SRC (e.g. RSBench_SRC).

***SYCL_EXTRA_TESTS_SRC*** path to directory which contains extra LIT tests.

It is asssumed that all required dependencies (OpenCL runtimes, CUDA SDK, AOT
compilers, etc) are available in the system.

See examples below for configuring tests targeting different devices:
 - Multiple backends iterative mode
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DCHECK_SYCL_ALL="PI_OPENCL:acc,gpu,cpu,host;PI_LEVEL_ZERO:gpu,host;PI_CUDA:gpu,host" -C../cmake/caches/clang_fsycl.cmake  ..
ninja -k0 check-sycl-all
```
 - SYCL host:
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_OPENCL -DSYCL_TARGET_DEVICES="host" -C../cmake/caches/clang_fsycl.cmake  ..
ninja check
```
 - OpenCL GPU
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_OPENCL -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl.cmake  ..
ninja check
```
 - OpenCL CPU
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_OPENCL -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl.cmake  ..
ninja check
```
 - OpenCL FPGA emulator
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_OPENCL -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl.cmake  ..
ninja check
```
 - CUDA GPU
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_CUDA -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl_cuda.cmake  ..
ninja check
```
 - Level Zero GPU
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_LEVEL_ZERO -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl.cmake  ..
ninja check
```

# LIT parameters accepted by LIT executor:
 * **dpcpp_compiler** - full path to dpcpp compiler;
 * **target_device** - comma-separated list of target devices (cpu, gpu, acc,
   host);
 * **sycl_be** - SYCL backend to be used (PI_OPENCL, PI_LEVEL_ZERO, PI_CUDA);
 * **dump_ir** - if IR dumping is supported for compiler (True, False);
 * **extra_environment** - comma-separated list of variables with values to be
   added to test environment. Can be also set by LIT_EXTRA_ENVIRONMENT variable
   in cmake.

# LIT features which can be used to configure test execution:
 * **windows**, **linux** - host OS;
 * **cpu**, **gpu**, **host**, **acc** - target devices;
 * **cuda**, **opencl**, **level_zero** - target backend;
 * **sycl-ls** - sycl-ls tool is available;
 * **dump_ir**: is set to true if compiler supports dumping IR. Can be also
   defined by setting DUMP_IR_SUPPORTED in cmake. Default values is false.
