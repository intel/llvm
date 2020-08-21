# Table of contents
 * [Overview](#overview)
 * [Execution](#execution)
 * [Directory structure](#directory-structure)
 * [CMake parameters](#cmake-parameters)
 * [LIT parameters accepted by LIT executor](#lit-parameters-accepted-by-lit-executor)
 * [LIT features which can be used to configure test execution](#lit-features-which-can-be-used-to-configure-test-execution)

# Overview
This directory contains SYCL-related infrastructure for running tests which
sources are stored outside of this repository. Tests are distributed in
sub-directories. One directory per application/test repository.

# Execution
```
git clone <GIT_REPO> # e.g. https://github.com/intel/llvm-test-suite
cd llvm-test-suite
mkdir build
cd build
# configuring test execution (selecting compiler, target BE and target device,
#downloading and building RSBench sources from default location)
cmake -G Ninja \
        -DTEST_SUITE_SUBDIRS=SYCL \
        -DTEST_SUITE_LIT=<PATH_TO_llvm-lit> \
        -DCHECK_SYCL_ALL="PI_OPENCL:cpu,acc,gpu,host;PI_LEVEL_ZERO:gpu,host" \
        -DCMAKE_CXX_COMPILER=<PATH_TO_sycl_compiler> \
        -DSYCL_EXTERNAL_TESTS="RSBench" \
        ..
# Build and run full list of SYCL tests
ninja check
# Build all tests dependencies
ninja
# Run all SYCL tests
llvm-lit .
# Get list of available tests
llvm-lit . --show-tests
# Run specific test (e.g. RSBench only)
llvm-lit SYCL/External/RSBench
# Run tests with parameters
llvm-lit --param target_devices=host,gpu --param sycl_be=PI_LEVEL_ZERO \
        --param dpcpp_compiler=path/to/clang++ --param dump_ir=True SYCL/External/RSBench
```

Notes:
 - it is assumed that LIT framework, FileCheck and other LIT dependencies are
available in the same directory with llvm-lit.

# Directory structure
Every sub-directory (e.g. RSBench) contains the following content:
 * **CMakeLists.txt** - CMake configuration file which is used to obtain
 application binary and data files. There are several variables which are
 used by it (see [Main parameters](#main-parameters) section for details)
 * **\*.test** - test configuration files, containing command lines, test
   status lines and verification patterns following [LLVM test infrastructure](https://llvm.org/docs/TestingGuide.html).
 * **lit.local.cfg** - application specific LIT infrastructure configuration
   file. In case of RSBench benchmarks it sets LIT test file suffixes to `.test`
   and mark all tests in directory unsupported if corresponding directory is
   not added to `SYCL_EXTERNAL_TESTS` list (benchmark is not included in
   testing).

# CMake parameters
All parameters described in [Readme.md](../README.md#cmake-parameters) are
applicable. Also extra CMake parameters are introduced to configure specific
application:
   * **APPName_BIN** (e.g. `RSBench_BIN`) - point to the directory containing
      prebuilt binaries of the application.
   * **APPName_SRC** (e.g. `RSBench_SRC`) - point to the directory containing
      sources of the application.
   * **APPName_URL** (e.g. `RSBench_URL`) - URL to the GIT repository containing
      sources of the application (default value for RSBench benchmark is
      `https://github.com/ANL-CESAR/RSBench.git`).
   * **APPName_TAG** (e.g. `RSBench_TAG`) - GIT tag or hash or branch name used
      to download source from GIT repository (default value for RSBench
      benchmark is `origin/master`).
Configuration parameters are priorities from top to down. If **APPName_BIN**
is specified binaries will be used directly ignoring other parameters.

# LIT parameters accepted by LIT executor:
All parameters described in [Readme.md](../README.md#lit-parameters-accepted-by-lit-executor) are applicable.

# LIT features which can be used to configure test execution:
All features described in [Readme.md](../README.md#lit-features-which-can-be-used-to-configure-test-execution) are applicable.
