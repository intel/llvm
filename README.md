# Unified Runtime

[![Build and test](https://github.com/oneapi-src/unified-runtime/actions/workflows/cmake.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/cmake.yml)
[![CodeQL](https://github.com/oneapi-src/unified-runtime/actions/workflows/codeql.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/codeql.yml)
[![Bandit](https://github.com/oneapi-src/unified-runtime/actions/workflows/bandit.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/bandit.yml)
[![Coverity](https://scan.coverity.com/projects/28213/badge.svg)](https://scan.coverity.com/projects/oneapi-src-unified-runtime)

## Adapters
Adapter implementations for Unified Runtime currently reside in the [SYCL repository](https://github.com/intel/llvm/tree/sycl/sycl/plugins/unified_runtime/ur). This branch contains scripts to automatically
fetch and build them directly in the UR tree. The adapters are disabled by default,
see cmake options for details.

<!-- TODO: add general description and purpose of the project -->

## Table of contents

- [Unified Runtime](#unified-runtime)
  - [Adapters](#adapters)
  - [Table of contents](#table-of-contents)
  - [Contents of the repo](#contents-of-the-repo)
  - [Integration](#integration)
    - [Weekly tags](#weekly-tags)
  - [Third-Party tools](#third-party-tools)
  - [Building](#building)
    - [Requirements](#requirements)
    - [Windows](#windows)
    - [Linux](#linux)
    - [CMake standard options](#cmake-standard-options)
    - [Additional make targets](#additional-make-targets)
  - [Contributions](#contributions)
    - [Adapter naming convention](#adapter-naming-convention)
    - [Source code generation](#source-code-generation)
    - [Documentation](#documentation)


## Contents of the repo

This repo contains the following:

- API specification in YaML
- API programming guide in RST
- Loader and a null adapter implementation (partially generated)
- Example applications
- API C/C++ header files (generated)
- API Python module (generated)
- Sample C++ wrapper (generated)
- Sample C/C++ import library (generated)

## Integration

The recommended way to integrate this project into another is via CMake's
[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html),
for example:

```cmake
include(FetchContent)

FetchContent_Declare(
    unified-runtime
    GIT_REPOSITORY https://github.com/oneapi-src/unified-runtime.git
    GIT_TAG main  # This will pull the latest changes from the main branch.
)
FetchContent_MakeAvailable(unified-runtime)

add_executable(example example.cpp)
target_link_libraries(example PUBLIC unified-runtime::headers)
```

### Weekly tags

Each Friday at 23:00 UTC time a [prerelease
tag](https://github.com/oneapi-src/unified-runtime/releases) is created which
takes the form `weekly-YYYY-MM-DD`. These tags should be used by downstream
projects which intend to track development closely but maintain a fixed point in
history to avoid pulling potentially breaking changes from the `main` branch.

## Third-Party tools

Tools can be acquired via instructions in [third_party](/third_party/README.md).

## Building

### Requirements

Required packages:
- C++ compiler with C++17 support
- [CMake](https://cmake.org/) >= 3.14.0
- Python v3.6.6 or later

For development and contributions:
- clang-format-15.0 (can be installed with `python -m pip install clang-format==15.0.7`)

### Windows

Generating Visual Studio Project. EXE and binaries will be in **build/bin/{build_config}**

```bash
$ mkdir build
$ cd build
$ cmake {path_to_source_dir} -G "Visual Studio 15 2017 Win64"
```

### Linux

Executable and binaries will be in **build/bin**

```bash
$ mkdir build
$ cd build
$ cmake {path_to_source_dir}
$ make
```

### CMake standard options

List of options provided by CMake:

| Name | Description | Values | Default |
| - | - | - | - |
| UR_BUILD_TESTS | Build the tests | ON/OFF | ON |
| UR_BUILD_TOOLS | Build tools | ON/OFF | ON |
| UR_FORMAT_CPP_STYLE | Format code style | ON/OFF | OFF |
| UR_DEVELOPER_MODE | Treat warnings as errors and enables additional checks | ON/OFF | OFF |
| UR_USE_ASAN | Enable AddressSanitizer | ON/OFF | OFF |
| UR_USE_TSAN | Enable ThreadSanitizer | ON/OFF | OFF |
| UR_USE_UBSAN | Enable UndefinedBehavior Sanitizer | ON/OFF | OFF |
| UR_USE_MSAN | Enable MemorySanitizer (clang only) | ON/OFF | OFF |
| UR_ENABLE_TRACING | Enable XPTI-based tracing layer | ON/OFF | OFF |
| UR_CONFORMANCE_TARGET_TRIPLES | SYCL triples to build CTS device binaries for | Comma-separated list | spir64 |
| UR_BUILD_ADAPTER_L0     | Fetch and use level-zero adapter from SYCL             | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_OPENCL | Fetch and use opencl adapter from SYCL                 | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_CUDA   | Fetch and use cuda adapter from SYCL                   | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_HIP    | Fetch and use hip adapter from SYCL                    | ON/OFF     | OFF     |
| UR_HIP_PLATFORM         | Build hip adapter for AMD or NVIDIA platform           | AMD/NVIDIA | AMD     |
| UR_ENABLE_COMGR         | Enable comgr lib usage           | AMD/NVIDIA | AMD     |
| UR_DPCXX | Path of the DPC++ compiler executable to build CTS device binaries | File path | `""` |
| UR_SYCL_LIBRARY_DIR | Path of the SYCL runtime library directory to build CTS device binaries | Directory path | `""` |

### Additional make targets

To run automated code formatting, configure CMake with `UR_FORMAT_CPP_STYLE` option
and then run a custom `cppformat` target:

```bash
$ make cppformat
```

If you've made modifications to the specification, you can also run
a custom `generate` target prior to building.
It will generate the source code **and** run automated code formatting:

```bash
$ make generate
```

## Contributions

For those who intend to make a contribution to the project please read our
[Contribution Guide](https://oneapi-src.github.io/unified-runtime/core/CONTRIB.html)
for more information.

### Adapter naming convention

To maintain consistency and clarity in naming adapter libraries, it is recommended
to use the following naming convention:

* On Linux platforms, use `libur_adapter_[name].so`.
* On Windows platforms, use `ur_adapter_[name].dll`.

### Source code generation

Code is generated using included [Python scripts](/scripts/README.md).

### Documentation

Documentation is generated from source code using Sphinx -
see [scripts dir](/scripts/README.md) for details.
