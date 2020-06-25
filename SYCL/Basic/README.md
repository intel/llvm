# Overview
SYCL related test based on SYCL-LIT. These tests support
execution on all supported devices and SYCL backends.

# Table of contents
 * [Execution](#execution)
 * [Main parameters](#main-parameters)
 * [LIT features which can be used to configure test execution](#lit-features-which-can-be-used-to-configure-test-execution)

# Execution
```
git clone <GIT_REPO> # e.g. https://github.com/vladimirlaz/llvm-test-suite
cd llvm-test-suite
mkdir build
cd build
# configuring test execution (selecting compiler version, target BE and target device)
cmake -G Ninja -DTEST_SUITE_SUBDIRS=SYCL -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=<SYCL_BE> -DSYCL_TARGET_DEVICES=<TARGET_DEVICES> -C<CMAKE_CHASHED_CONFIG> ..
# Building full list of tests in subdir
ninja check
# or
llvm-lit .
# Get list of available tests
llvm-lit . --show-tests
# Run specific test
llvm-lit <path_to_test>
# Run tests with parameters
llvm-lit --param target_devices=host,gpu --param sycl_be=PI_LEVEL0 --param dpcpp_compiler=path/to/clang++ --param dump_ir=True .
```

Notes:
 - it is assumed that LIT framework, FileCheck and other LIT dependencies are available in the same directory with llvm-lit.
 - compiler variant as well as compile/link options are defined in cashed cmake configurations:
   - [dpcpp.cmake](../../cmake/caches/dpcpp.cmake)
   - [clang_fsycl.cmake](../../cmake/cashes/clang_fsycl.cmake)
   - [clang_fsycl_cuda.cmake](../../cmake/cashes/clang_fsycl_cuda.cmake)
 - compiler is taken from environment.

# Main parameters
It is possible to change tets scope my specifying test directory/file in first
argument to for thelit-runner.py script.

***SYCL_TARGET_DEVICES*** should point to the directory containing DPCPP compiler

***SYCL_TARGET_DEVICES*** defines comma separated target device types (default value is
 cpu,gpu,acc,host). Supported target_devices values are:
 - **cpu**  - CPU device available in OpenCL backend only;
 - **gpu**  - GPU device available in OpenCL, Level Zero and CUDA backends;
 - **acc**  - FPGA emulator device available in OpenCL backend only;
 - **host** - SYCL Host device availabel with all backends.

***SYCL_BE*** defined SYCL backend to be used for testing (default is PI_OPENCL).
Supported sycl_be values:
 - PI_OPENCL - for OpenCL backend;
 - PI_CUDA - for CUDA backend;
 - PI_LEVEL0 - Level Zero backend.

It is asssumed that all dependencies (OpenCL runtimes,
CUDA SDK, AOT compilers, etc) are available in the system.

See examples below for configuring tests targetting different devices:
 - SYCL host:
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_OPENCL -DSYCL_TARGET_DEVICES="host" -C../cmake/caches/clang_fsycl.cmake  ..
```
 - OpenCL GPU
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_OPENCL -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl.cmake  ..
```
 - OpenCL CPU
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_OPENCL -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl.cmake  ..
```
 - OpenCL FPGA emulator
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_OPENCL -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl.cmake  ..
```
 - CUDA GPU
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_CUDA -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl_cuda.cmake  ..
```
 - Level Zero GPU
```
cmake -G Ninja  -DTEST_SUITE_COLLECT_CODE_SIZE=OFF  -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF -DTEST_SUITE_SUBDIRS=SYCL  -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> -DSYCL_BE=PI_LEVEL0 -DSYCL_TARGET_DEVICES="gpu" -C../cmake/caches/clang_fsycl.cmake  ..
```

# LIT parameters can be passed to LIT executor:
 - **dpcpp_compiler** - full path to dpcpp compiler;
 - **target_device** - comma separated list of target devices (cpu, gpu, acc, host);
 - **sycl_be** - SYCL backedn to be used (PI_OPENCL, PI_LEVEL, PI_CUDA);
 - **dump_ir** - if IR dumping is supported for compiler (True, False).

# LIT features which can be used to configure test execution:
 - **windows**, **linux** - host OS;
 - **cpu**, **gpu**, **host**, **acc** - target devices;
 - **cuda**, **opencl**, **level0** - target backend;
 - **sycl-ls** - sycl-ls tool is available;
 - **dump_ir**: is set to true if compiler supports dumiping IR. Can be set by setting DUMP_IR_SUPPORTED in cmake. Default is false.

