# Unified Runtime

[![GHA build status](https://github.com/oneapi-src/unified-runtime/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/oneapi-src/unified-runtime/actions)

## Adapters
Adapter implementations for Unified Runtime currently reside in the [SYCL repository](https://github.com/intel/llvm/tree/sycl/sycl/plugins/unified_runtime/ur). This branch contains scripts to automatically
fetch and build them directly in the UR tree. The adapters are disabled by default,
see cmake options for details.

> **Note**
>
> For those who intend to make a contribution to the project please read our
> [Contribution Guide](https://oneapi-src.github.io/unified-runtime/core/CONTRIB.html)
> for more information.

## Contents

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

### Weekly Tags

Each Friday at 23:00 UTC time a [prerelease
tag](https://github.com/oneapi-src/unified-runtime/releases) is created which
takes the form `weekly-YYYY-MM-DD`. These tags should be used by downstream
projects which intend to track development closely but maintain a fixed point in
history to avoid pulling potentially breaking changes from the `main` branch.

## Source Code Generation

Code is generated using included [Python scripts](/scripts/README.md).

## Documentation

Documentation is generated from source code using Sphinx.

## Third-Party Tools

Tools can be acquired via instructions in [third_party](/third_party/README.md).

## Building

Requirements:
- C++ compiler with C++17 support
- cmake >= 3.14.0
- clang-format-15.0 (can be installed with `python -m pip install clang-format==15.0.7`)

Project is defined using [CMake](https://cmake.org/).

**Windows**:

Generating Visual Studio Project.  EXE and binaries will be in **build/bin/{build_config}**

~~~~
$ mkdir build
$ cd build
$ cmake {path_to_source_dir} -G "Visual Studio 15 2017 Win64"
~~~~

**Linux**:

Executable and binaries will be in **build/bin**

~~~~
$ mkdir build
$ cd build
$ cmake {path_to_source_dir}
$ make
~~~~

### CMake standard options

List of options provided by CMake:

| Name | Description | Values | Default |
| - | - | - | - |
| UR_BUILD_TESTS | Build the tests | ON/OFF | ON |
| UR_FORMAT_CPP_STYLE | Format code style | ON/OFF | OFF |
| UR_DEVELOPER_MODE | Treat warnings as errors and enables additional checks | ON/OFF | OFF |
| UR_USE_ASAN | Enable AddressSanitizer | ON/OFF | OFF |
| UR_USE_UBSAN | Enable UndefinedBehavior Sanitizer | ON/OFF | OFF |
| UR_USE_MSAN | Enable MemorySanitizer (clang only) | ON/OFF | OFF |
| UR_ENABLE_TRACING | Enable XPTI-based tracing layer | ON/OFF | OFF |
| UR_BUILD_TOOLS | Build tools | ON/OFF | ON |
| UR_BUILD_ADAPTER_L0 | Fetch and use level-zero adapter from SYCL | ON/OFF | OFF |
| UR_BUILD_ADAPTER_CUDA | Fetch and use cuda adapter from SYCL | ON/OFF | OFF |
| UR_BUILD_ADAPTER_HIP | Fetch and use hip adapter from SYCL | ON/OFF | OFF |
| UR_HIP_PLATFORM | Build hip adapter for AMD or NVIDIA platform | AMD/NVIDIA | AMD |

**General**:

To run automated code formatting build with option `UR_FORMAT_CPP_STYLE` and then run a custom `cppformat` target:
~~~~
$ make cppformat
~~~~

If you've made modifications to the specification, you can also run a custom `generate-code` or `generate` target prior to building.
This call will generate the source code:
~~~~
$ make generate-code
~~~~

This call will generate the source code and run automated code formatting:
~~~~
$ make generate
~~~~
Note: `generate` target requires `UR_FORMAT_CPP_STYLE` to bet set.

### Adapter naming convention

To maintain consistency and clarity in naming adapter libraries, it is recommended
to use the following naming convention:

* On Linux platforms, use `libur_adapter_[name].so`.
* On Windows platforms, use `ur_adapter_[name].dll`.
