# Unified Runtime

## Contents

This repo contains the following:

- API specification in YaML
- API programming guide in RST
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

## Source Code Generation

Code is generated using included [Python scripts](/scripts/README.md).  

## Documentation

Documentation is generated from source code using Sphinx.

## Third-Party Tools

Tools can be acquired via instructions in [third_party](/third_party/README.md).

## Building

Project is defined using [CMake](https://cmake.org/).

**Windows**:
Generating Visual Studio Project.  EXE and binaries will be in **build/bin/{build_config}**

~~~~
mkdir build
cd build
cmake {path_to_source_dir} -G "Visual Studio 15 2017 Win64"
~~~~

**Linux**:

Executable and binaries will be in **build/bin**

~~~~
mkdir build
cd build
cmake {path_to_source_dir}
make
~~~~

