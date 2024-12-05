# Unified Runtime

[![Build and test](https://github.com/oneapi-src/unified-runtime/actions/workflows/cmake.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/cmake.yml)
[![Bandit](https://github.com/oneapi-src/unified-runtime/actions/workflows/bandit.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/bandit.yml)
[![CodeQL](https://github.com/oneapi-src/unified-runtime/actions/workflows/codeql.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/codeql.yml)
[![Coverity build](https://github.com/oneapi-src/unified-runtime/actions/workflows/coverity.yml/badge.svg?branch=main)](https://github.com/oneapi-src/unified-runtime/actions/workflows/coverity.yml)
[![Coverity report](https://scan.coverity.com/projects/28213/badge.svg)](https://scan.coverity.com/projects/oneapi-src-unified-runtime)
[![Nightly](https://github.com/oneapi-src/unified-runtime/actions/workflows/nightly.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/nightly.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/oneapi-src/unified-runtime/badge)](https://securityscorecards.dev/viewer/?uri=github.com/oneapi-src/unified-runtime)
[![Trivy](https://github.com/oneapi-src/unified-runtime/actions/workflows/trivy.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/trivy.yml)
[![Deploy documentation to Pages](https://github.com/oneapi-src/unified-runtime/actions/workflows/docs.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/docs.yml)
[![Compute Benchmarks Nightly](https://github.com/oneapi-src/unified-runtime/actions/workflows/benchmarks-nightly.yml/badge.svg)](https://github.com/oneapi-src/unified-runtime/actions/workflows/benchmarks-nightly.yml)

<!-- TODO: add general description and purpose of the project -->

## Table of contents

- [Unified Runtime](#unified-runtime)
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
  - [Release Process](#release-process)

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

The requirements and instructions below are for building the project from source
without any modifications. To make modifications to the specification, please
see the
[Contribution Guide](https://oneapi-src.github.io/unified-runtime/core/CONTRIB.html)
for more detailed instructions on the correct setup.

### Requirements

Required packages:
- C++ compiler with C++17 support
- [CMake](https://cmake.org/) >= 3.20.0
- Python v3.6.6 or later

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
| UR_BUILD_EXAMPLES | Build example applications | ON/OFF | ON |
| UR_BUILD_TESTS | Build the tests | ON/OFF | ON |
| UR_BUILD_TOOLS | Build tools | ON/OFF | ON |
| UR_FORMAT_CPP_STYLE | Format code style | ON/OFF | OFF |
| UR_DEVELOPER_MODE | Treat warnings as errors | ON/OFF | OFF |
| UR_ENABLE_FAST_SPEC_MODE | Enable fast specification generation mode | ON/OFF | OFF |
| UR_USE_ASAN | Enable AddressSanitizer | ON/OFF | OFF |
| UR_USE_TSAN | Enable ThreadSanitizer | ON/OFF | OFF |
| UR_USE_UBSAN | Enable UndefinedBehavior Sanitizer | ON/OFF | OFF |
| UR_USE_MSAN | Enable MemorySanitizer (clang only) | ON/OFF | OFF |
| UR_USE_CFI | Enable Control Flow Integrity checks (clang only, also enables lto) | ON/OFF | OFF |
| UR_ENABLE_TRACING | Enable XPTI-based tracing layer | ON/OFF | OFF |
| UR_ENABLE_SANITIZER | Enable device sanitizer layer | ON/OFF | ON |
| UR_CONFORMANCE_TARGET_TRIPLES | SYCL triples to build CTS device binaries for | Comma-separated list | spir64 |
| UR_CONFORMANCE_AMD_ARCH | AMD device target ID to build CTS binaries for | string | `""` |
| UR_CONFORMANCE_ENABLE_MATCH_FILES | Enable CTS match files | ON/OFF | ON |
| UR_CONFORMANCE_TEST_LOADER | Additionally build and run "loader" tests for the CTS | ON/OFF | OFF |
| UR_BUILD_ADAPTER_L0     | Build the Level-Zero adapter            | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_OPENCL | Build the OpenCL adapter                | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_CUDA   | Build the CUDA adapter                  | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_HIP    | Build the HIP adapter                   | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_NATIVE_CPU | Build the Native-CPU adapter        | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_ALL    | Build all currently supported adapters  | ON/OFF     | OFF     |
| UR_BUILD_ADAPTER_L0_V2    | Build the (experimental) Level-Zero v2 adapter  | ON/OFF     | OFF     |
| UR_STATIC_ADAPTER_L0    | Build the Level-Zero adapter as static and embed in the loader | ON/OFF   | OFF |
| UR_HIP_PLATFORM         | Build HIP adapter for AMD or NVIDIA platform           | AMD/NVIDIA | AMD     |
| UR_ENABLE_COMGR         | Enable comgr lib usage           | AMD/NVIDIA | AMD     |
| UR_DPCXX | Path of the DPC++ compiler executable to build CTS device binaries | File path | `""` |
| UR_DEVICE_CODE_EXTRACTOR | Path of the `clang-offload-extract` executable from the DPC++ package, required for CTS device binaries | File path | `"${dirname(UR_DPCXX)}/clang-offload-extract"` |
| UR_DPCXX_BUILD_FLAGS | Build flags to pass to DPC++ when compiling device programs | Space-separated options list | `""` |
| UR_SYCL_LIBRARY_DIR | Path of the SYCL runtime library directory to build CTS device binaries | Directory path | `""` |
| UR_HIP_ROCM_DIR | Path of the default ROCm HIP installation | Directory path | `$ENV{ROCM_PATH}` or `/opt/rocm` |
| UR_HIP_INCLUDE_DIR | Path of the ROCm HIP include directory | Directory path | `${UR_HIP_ROCM_DIR}/include` |
| UR_HIP_HSA_INCLUDE_DIRS | Path of the ROCm HSA include directory | Directory path | `${UR_HIP_ROCM_DIR}/hsa/include;${UR_HIP_ROCM_DIR}/include` |
| UR_HIP_LIB_DIR | Path of the ROCm HIP library directory | Directory path | `${UR_HIP_ROCM_DIR}/lib` |

### Additional make targets

To run tests, do the following:

```bash
$ make
$ make test
```

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

This target has additional dependencies which are described in the *Build
Environment* section of the
[Contribution Guide](https://oneapi-src.github.io/unified-runtime/core/CONTRIB.html).

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

## Release Process

Unified Runtime releases are aligned with oneAPI releases. Once all changes
planned for a release have been accepted, the release process is defined as:

1. Create a new release branch based on the [main][main-branch] branch taking
   the form `v<major>.<minor>.x` where `x` is a placeholder for the patch
   version. This branch will always contain the latest patch version for a given
   release.
2. Create a PR to increment the CMake project version on the [main][main-branch]
   and merge before accepting any other changes.
3. Create a new tag based on the latest commit on the release branch taking the
   form `v<major>.<minor>.<patch>`.
4. Create a [new GitHub release][new-github-release] using the tag created in
   the previous step.
   * Prior to version 1.0, check the *Set as a pre-release* tick box.
5. Update downstream projects to utilize the release tag. If any issues arise
   from integration, apply any necessary hot fixes to `v<major>.<minor>.x`
   branch and go back to step 3.

[main-branch]: https://github.com/oneapi-src/unified-runtime/tree/main
[new-github-release]: https://github.com/oneapi-src/unified-runtime/releases/new
