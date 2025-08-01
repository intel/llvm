# Getting Started with oneAPI DPC++

The DPC++ Compiler compiles C++ and SYCL\* source files with code for both CPU
and a wide range of compute accelerators such as GPU and FPGA.

## Table of contents

  * [Prerequisites](#prerequisites)
    * [Create DPC++ workspace](#create-dpc-workspace)
  * [Build DPC++ toolchain](#build-dpc-toolchain)
    * [Build DPC++ toolchain with libc++ library](#build-dpc-toolchain-with-libc-library)
    * [Build DPC++ toolchain with support for NVIDIA CUDA](#build-dpc-toolchain-with-support-for-nvidia-cuda)
    * [Build DPC++ toolchain with support for HIP AMD](#build-dpc-toolchain-with-support-for-hip-amd)
    * [Build DPC++ toolchain with support for HIP NVIDIA](#build-dpc-toolchain-with-support-for-hip-nvidia)
    * [Build DPC++ toolchain with support for ARM processors](#build-dpc-toolchain-with-support-for-arm-processors)
    * [Build DPC++ toolchain with additional features enabled that require runtime/JIT compilation](#build-dpc-toolchain-with-additional-features-enabled-that-require-runtimejit-compilation)
    * [Build DPC++ toolchain with a custom Unified Runtime](#build-dpc-toolchain-with-a-custom-unified-runtime)
    * [Build DPC++ toolchain with device image compression support](#build-dpc-toolchain-with-device-image-compression-support)
    * [Build Doxygen documentation](#build-doxygen-documentation)
    * [Deployment](#deployment)
  * [Use DPC++ toolchain](#use-dpc-toolchain)
    * [Install low level runtime](#install-low-level-runtime)
    * [Obtain prerequisites for ahead of time (AOT) compilation](#obtain-prerequisites-for-ahead-of-time-aot-compilation)
      * [GPU](#gpu)
      * [CPU](#cpu)
      * [Accelerator](#accelerator)
    * [Test DPC++ toolchain](#test-dpc-toolchain)
      * [Run in-tree LIT tests](#run-in-tree-lit-tests)
      * [Run DPC++ E2E tests](#run-dpc-e2e-tests)
      * [Run Khronos\* SYCL\* conformance test suite (optional)](#run-khronos-sycl-conformance-test-suite-optional)
    * [Run simple DPC++ application](#run-simple-dpc-application)
      * [AOT Target architectures](#aot-target-architectures)
    * [Build DPC++ application with CMake](#build-dpc-application-with-cmake)
    * [Code the program for a specific GPU](#code-the-program-for-a-specific-gpu)
  * [C++ standard](#c-standard)
  * [Known Issues and Limitations](#known-issues-and-limitations)
    * [CUDA back-end limitations](#cuda-back-end-limitations)
    * [HIP back-end limitations](#hip-back-end-limitations)
  * [Find More](#find-more)

## Prerequisites

| Software                                                                    | Version                                                                                                                              |
| ---                                                                         | ---                                                                                                                                  |
| [Git](https://git-scm.com/downloads)                                        |                                                                                                                                      |
| [CMake](http://www.cmake.org/download/)                                     | [See LLVM](https://github.com/intel/llvm/blob/sycl/llvm/docs/GettingStarted.rst#software)                                            |
| [Python](https://www.python.org/downloads/)                                 | [See LLVM](https://github.com/intel/llvm/blob/sycl/llvm/docs/GettingStarted.rst#software)                                            |
| [Ninja](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages) |                                                                                                                                      |
| `hwloc`                                                                     | >= 2.3 (Linux only, `libhwloc-dev` or `hwloc-devel`)                                                                                 |
| C++ compiler                                                                | [See LLVM](https://github.com/intel/llvm/blob/sycl/llvm/docs/GettingStarted.rst#host-c-toolchain-both-compiler-and-standard-library) |
|`zstd` (optional) | >= 1.4.8 (see [ZSTD](#build-dpc-toolchain-with-device-image-compression-support)) |

Alternatively, you can create a Docker image that has everything you need for
building pre-installed using the [Ubuntu 24.04 build Dockerfile](https://github.com/intel/llvm/blob/sycl/devops/containers/ubuntu2404_build.Dockerfile).

See [Docker BKMs](developer/DockerBKMs.md) for more info on
Docker commands.

### Create DPC++ workspace

Throughout this document `DPCPP_HOME` denotes the path to the local directory
created as DPC++ workspace. It might be useful to create an environment variable
with the same name.

**Linux**:

```bash
export DPCPP_HOME=~/sycl_workspace
mkdir $DPCPP_HOME
cd $DPCPP_HOME

git clone https://github.com/intel/llvm -b sycl
```

**Windows (64-bit)**:

Open a developer command prompt using one of two methods:

* Click start menu and search for "**x64** Native Tools Command Prompt for VS
  XXXX", where XXXX is a version of installed Visual Studio.
* Ctrl-R, write "cmd", click enter, then run
  `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64`

```bat
set DPCPP_HOME=%USERPROFILE%\sycl_workspace
mkdir %DPCPP_HOME%
cd %DPCPP_HOME%

git clone --config core.autocrlf=false https://github.com/intel/llvm -b sycl
```

## Build DPC++ toolchain

The easiest way to get started is to use the buildbot
[configure](https://github.com/intel/llvm/blob/sycl/buildbot/configure.py) and
[compile](https://github.com/intel/llvm/blob/sycl/buildbot/compile.py) scripts.

In case you want to configure CMake manually the up-to-date reference for
variables is in these files. Note that the CMake variables set by default by the [configure.py](../../buildbot/configure.py) script are the ones commonly used by
 DPC++ developers and might not necessarily suffice for your project-specific needs.

**Linux**:

```bash
python $DPCPP_HOME/llvm/buildbot/configure.py
python $DPCPP_HOME/llvm/buildbot/compile.py
```

**Windows (64-bit)**:

```bat
python %DPCPP_HOME%\llvm\buildbot\configure.py
python %DPCPP_HOME%\llvm\buildbot\compile.py
```

You can use the following flags with `configure.py` (full list of available
flags can be found by launching the script with `--help`):

* `--werror` -> treat warnings as errors when compiling LLVM
* `--cuda` -> use the cuda backend (see
  [Nvidia CUDA](#build-dpc-toolchain-with-support-for-nvidia-cuda))
* `--hip` -> use the HIP backend (see
  [HIP](#build-dpc-toolchain-with-support-for-hip-amd))
* `--hip-platform` -> select the platform used by the hip backend, `AMD` or
  `NVIDIA` (see [HIP AMD](#build-dpc-toolchain-with-support-for-hip-amd) or see
  [HIP NVIDIA](#build-dpc-toolchain-with-support-for-hip-nvidia))
* `--enable-all-llvm-targets` -> build compiler (but not a runtime) with all
  supported targets
* `--shared-libs` -> Build shared libraries
* `-t` -> Build type (Debug or Release)
* `-o` -> Path to build directory
* `--cmake-gen` -> Set build system type (e.g. `--cmake-gen "Unix Makefiles"`)
* `--use-zstd` -> Force link zstd while building LLVM (see [ZSTD](#build-dpc-toolchain-with-device-image-compression-support))

You can use the following flags with `compile.py` (full list of available flags
can be found by launching the script with `--help`):

* `-o` -> Path to build directory
* `-t`, `--build-target` -> Build target (e.g., `clang` or `llvm-spirv`).
  Default is `deploy-sycl-toolchain`
* `-j`, `--build-parallelism` -> Number of threads to use for compilation

**Please note** that no data about flags is being shared between `configure.py`
and `compile.py` scripts, which means that if you configured your build to be
placed in non-default directory using `-o` flag, you must also specify this flag
and the same path in `compile.py` options. This allows you, for example, to
configure several different builds and then build just one of them which is
needed at the moment.

### Build DPC++ toolchain with libc++ library

There is experimental support for building and linking DPC++ runtime with libc++
library instead of libstdc++. To enable it the following CMake option should be
used.

**Linux**:

```sh
-DLLVM_ENABLE_LIBCXX=ON
```

You can also use configure script to enable:

```sh
python %DPCPP_HOME%\llvm\buildbot\configure.py --use-libcxx
python %DPCPP_HOME%\llvm\buildbot\compile.py
```

### Build DPC++ toolchain with support for NVIDIA CUDA

To enable support for CUDA devices, follow the instructions for the Linux or
Windows DPC++ toolchain, but add the `--cuda` flag to `configure.py`. Note, the
CUDA backend has Windows support; Windows Subsystem for Linux (WSL) is not
needed to build and run the CUDA backend.

Refer to
[NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
or
[NVIDIA CUDA Installation Guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
for CUDA toolkit installation instructions.

Errors may occur if DPC++ is built with a toolkit version which is higher than
the CUDA driver version. In order to check that the CUDA driver and toolkits
match, use the CUDA executable `deviceQuery` which is usually found in
`$CUDA_INSTALL_DIR/cuda/extras/demo_suite/deviceQuery`.

**_NOTE:_** An installation of at least
[CUDA 11.6](https://developer.nvidia.com/cuda-downloads) is recommended because
there is a known issue with some math built-ins when using -O1/O2/O3
Optimization options for CUDA toolkits prior to 11.6 (This is due to a bug in
earlier versions of the CUDA toolkit: see
[this issue](https://forums.developer.nvidia.com/t/libdevice-functions-causing-ptxas-segfault/193352)).

**_NOTE:_** CUDA toolkit versions earlier than 11.0 are not regularly tested,
but should work for appropriate devices. Note that for example some oneapi
extensions that require sm_80 and later architectures also require at least CUDA
11.0.

The CUDA backend should work on Windows or Linux operating systems with any GPU
with compute capability (SM version) sm_50 or above. The default SM version for
the NVIDIA CUDA backend is sm_50. Users of sm_3X devices can attempt to specify
the target architecture [ahead of time](#aot-target-architectures), provided
that they use a 11.X  or earlier CUDA toolkit version, but some features may not be
supported. The CUDA backend has been tested with different Ubuntu Linux
distributions and a selection of supported CUDA toolkit versions and GPUs.
The backend is tested by a relevant device/toolkit prior to a ONEAPI plugin release.
Go to the plugin release
[pages](https://developer.codeplay.com/products/oneapi/nvidia/) for further
details.


**Non-standard CUDA location**:

If the CUDA toolkit is installed in a non-default location on your system, two
considerations must be made.

Firstly, **do not** add the toolkit to your standard environment variables
(`PATH`, `LD_LIBRARY_PATH`), as to do so will create conflicts with OpenCL
headers.

Secondly pass the CMake variable `CUDAToolkit_ROOT` as follows:

```sh
CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/configure.py \
    --cuda -DCUDA_Toolkit_ROOT=/path/to/cuda/toolkit

CC=gcc CXX=g++ python $DPCPP_HOME/llvm/buildbot/compile.py

$DPCPP_HOME/llvm/build/bin/clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/path/to/cuda/toolkit *.cpp -o a.out

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_HOME/llvm/build/lib ./a.out
```

### Build DPC++ toolchain with support for HIP AMD

To enable support for HIP devices, follow the instructions for the Linux DPC++
toolchain, but add the `--hip` flag to `configure.py`.

Enabling this flag requires an installation of ROCm on the system, for
instruction on how to install this refer to
[AMD ROCm Installation Guide for Linux](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

ROCm versions above 5.7 are recommended as earlier versions don't have graph
support. DPC++ aims to support new ROCm versions as they come out, so there may
be a delay but generally the latest ROCm version should work. The ROCm support
is mostly tested on AMD Radeon Pro W6800 (gfx1030), and MI250x (gfx90a), however
other architectures supported by LLVM may work just fine. The full list of ROCm
versions tested prior to oneAPI releases are listed on the plugin release
[pages](https://developer.codeplay.com/products/oneapi/amd).

The DPC++ build assumes that ROCm is installed in `/opt/rocm`, if it is
installed somewhere else, the directory must be provided through the CMake
variable `UR_HIP_ROCM_DIR` which can be passed through to cmake using the
configure helper script as follows:

```sh
python $DPCPP_HOME/llvm/buildbot/configure.py --hip \
  -DUR_HIP_ROCM_DIR=/usr/local/rocm
```
If further customization is required — for instance when the layout of
individual directories can not be inferred from `UR_HIP_ROCM_DIR` —
it is possible to specify the location of HIP include, HSA include and HIP
library directories, using the following CMake variables:
* `UR_HIP_INCLUDE_DIR`,
* `UR_HIP_HSA_INCLUDE_DIR`,
* `UR_HIP_LIB_DIR`.
These options are all passed through to Unified Runtime, more detail about them
can be found [here](https://github.com/oneapi-src/unified-runtime#cmake-standard-options).

[LLD](https://llvm.org/docs/AMDGPUUsage.html) is necessary for the AMDGPU
compilation chain. The AMDGPU backend generates a standard ELF relocatable code
object that can be linked by lld to produce a standard ELF shared code object
which can be loaded and executed on an AMDGPU target. The LLD project is enabled
by default when configuring for HIP. For more details on building LLD refer to
[LLD Build Guide](https://lld.llvm.org/).

### Build DPC++ toolchain with support for HIP NVIDIA

HIP applications can be built to target Nvidia GPUs, so in theory it is possible
to build the DPC++ HIP support for Nvidia, however this is not supported, so it
may not work.

There is no continuous integration for this and there are no guarantees for
supported platforms or configurations.

This is a compatibility feature and the
[CUDA backend](#build-dpc-toolchain-with-support-for-nvidia-cuda)
should be preferred to run on NVIDIA GPUs.

To enable support for HIP NVIDIA devices, follow the instructions for the Linux
DPC++ toolchain, but add the `--hip` and `--hip-platform NVIDIA` flags to
`configure.py`.

Enabling this flag requires HIP to be installed, specifically for Nvidia, see
the Nvidia tab on the HIP installation docs
[here](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html),
as well as the CUDA Runtime API to be installed, see [NVIDIA CUDA Installation
Guide for
Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### Build DPC++ toolchain with support for ARM processors

There is no continuous integration for this, and there are no guarantees for supported platforms or configurations.

To enable support for ARM processors, follow the instructions for the linux
toolchain, but add the `--host-target "AArch64;ARM;X86"` flag to `configure.py`.

For CUDA support also add the `--cuda` flag.

Currently, this has only been tried on Linux, with CUDA 12.3, and using a 
Grace Hopper platform with a ARM64 processor and an H100 GPU.

### Build DPC++ toolchain with additional features enabled that require runtime/JIT compilation

Support for runtime compilation of SYCL source code (via the
`sycl_ext_oneapi_kernel_compiler` extension) is enabled by default. The same
mechanism is used to allow JIT compilation of AMD and Nvidia kernels, including
materialization of specialization constants.

To disable support for these features, add the `--disable-jit` flag.

JIT compilation of AMD and Nvidia kernels is not yet supported on the Windows
platform.

### Build DPC++ toolchain with device image compression support

Device image compression enables the compression of device code (SYCL Kernels) during compilation and decompressing them on-demand during the execution of the corresponding SYCL application.
This reduces the size of fat binaries for both Just-in-Time (JIT) and Ahead-of-Time (AOT) compilation. Refer to the [blog post](https://www.intel.com/content/www/us/en/developer/articles/technical/sycl-compilation-device-image-compression.html) for more details on this feature.

To enable device image compression, you need to build the DPC++ toolchain with the
zstd compression library. By default, zstd is optional for DPC++ builds i.e. CMake will search for zstd installation but if not found, it will not fail the build
and this feature will simply be disabled.

To override this behavior and force the build to use zstd, you can use the `--use-zstd` flag in the `configure.py` script or by adding `-DLLVM_ENABLE_ZSTD=FORCE_ON` to the CMake configuration command.

#### How to obtain zstd?

Minimum zstd version that we have tested with is *1.4.8*.

**Linux**:

You can install zstd using the package manager of your distribution. For example, on Ubuntu, you can run:
```sh
sudo apt-get install libzstd-dev
```
Note that the libzstd-dev package provided on Ubuntu 24.04 has a bug ([link](https://bugs.launchpad.net/ubuntu/+source/libzstd/+bug/2086543)) and the zstd static library is not built with the `-fPIC` flag. Linking to this library will result in a build failure. For example: [Issue#15935](https://github.com/intel/llvm/issues/15935). As an alternative, zstd can be built from source either manually or by using the [build_zstd_1_5_6_ub24.sh](https://github.com/intel/llvm/blob/sycl/devops/scripts/build_zstd_1_5_6_ub24.sh) script.

**Windows**

For Windows, prebuilt zstd binaries can be obtained from the [facebook/zstd](https://github.com/facebook/zstd/releases/tag/v1.5.6) release page. After obtaining the zstd binaries, you can add the path to the zstd installation directory to the `PATH` environment variable.

### Build Doxygen documentation

Building Doxygen documentation is similar to building the product itself. First,
the following tools need to be installed:

* doxygen
* graphviz
* sphinx

Then you'll need to add the following options to your CMake configuration
command:

```sh
-DLLVM_ENABLE_DOXYGEN=ON
```

After CMake cache is generated, build the documentation with `doxygen-sycl`
target. It will be put to `$DPCPP_HOME/llvm/build/tools/sycl/doc/html`
directory.

### Build DPC++ toolchain with a custom Unified Runtime

DPC++ uses the [Unified Runtime](https://github.com/oneapi-src/unified-runtime)
under the hood to provide implementations of various SYCL backends. By default
the source code for the Unified Runtime will be acquired using CMake's
[FetchCotent](https://cmake.org/cmake/help/latest/module/FetchContent.html). The
specific repository URL and revision tag used can be found in the file
`sycl/cmake/modules/FetchUnifiedRuntime.cmake` searching for the variables
`UNIFIED_RUNTIME_REPO` and `UNIFIED_RUNTIME_TAG`.

In order to enable developers, a number of CMake variables are available to
control which revision of Unified Runtime should be used when building DPC++:

* `SYCL_UR_OVERRIDE_FETCH_CONTENT_REPO` is a variable which can be used to
  override the `UNIFIED_RUNTIME_REPO` variable used by `FetchContent` to attain
  the Unified Runtime source code.
* `SYCL_UR_OVERRIDE_FETCH_CONTENT_TAG` is a variable which can be used to
  override the `UNIFIED_RUNTIME_TAG` variable used by `FetchContent` to attain
  the Unified Runtime source code.
* `SYCL_UR_USE_FETCH_CONTENT` is an option to control if CMake should use
  `FetchContent` to pull in the Unified Runtime repository, it defaults to `ON`.
  When set to `OFF`, `FetchContent` will not be used, instead:
  * The path specified by variable `SYCL_UR_SOURCE_DIR` will be used with
    `add_directory()`. This can be used to point at an adjacent directory
    containing a clone of the Unified Runtime repository.
  * The path `sycl/unified-runtime` will be used, if it
    exists. This can be used as-if an in-tree build.
* `SYCL_UR_SOURCE_DIR` is a variable used to specify the path to the Unified
  Runtime repository when `SYCL_UR_USE_FETCH_CONTENT` is set of `OFF`.

### Build DPC++ libclc with a custom toolchain

libclc is an implementation of the OpenCL required libraries, as described in
the [OpenCL C specification](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_C.html),
additionally providing definitions of SPIR-V builtins. It is built to
target-specific bitcode, that is linked against SYCL binaries. By default, the
built system uses the SYCL toolchain currently being built to create libclc
bitcode. This can be suboptimal in case of debug builds, in which case debug
tools are used to build non-debug libclc bitcode (the notion of debug builds
doesn't really apply to libclc), resulting in very long compilation time. In
order to specify a directory containing custom toolchain users can set:
`LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR` variable. Care is required, as the
changes to the local SYCL tree might not be reflected in the custom location
during the build time.

### Deployment

TODO: add instructions how to deploy built DPC++ toolchain.

## Use DPC++ toolchain

### Install low level runtime

To run DPC++ applications on OpenCL devices, OpenCL implementation(s) must be
present in the system.

To run DPC++ applications on Level Zero devices, Level Zero implementation(s)
must be present in the system. You can find the link to the Level Zero spec in
the following section [Find More](#find-more).

The Level Zero RT for `GPU`, OpenCL RT for `GPU`, OpenCL RT for `CPU`, FPGA
emulation RT and TBB runtime which are needed to run DPC++ application
on Intel `GPU` or Intel `CPU` devices can be downloaded using links in
[the dependency configuration file](../../devops/dependencies.json)
and installed following the instructions below. The same versions are used in
PR testing.

**Linux**:

1) Extract the archive. For example, for the archives
`oclcpuexp_<cpu_version>.tar.gz` and `fpgaemu_<fpga_version>.tar.gz` you would
run the following commands

    ```bash
    # Extract OpenCL FPGA emulation RT
    mkdir -p /opt/intel/oclfpgaemu_<fpga_version>
    cd /opt/intel/oclfpgaemu_<fpga_version>
    tar zxvf fpgaemu_<fpga_version>.tar.gz
    # Extract OpenCL CPU RT
    mkdir -p /opt/intel/oclcpuexp_<cpu_version>
    cd /opt/intel/oclcpuexp_<cpu_version>
    tar -zxvf oclcpuexp_<cpu_version>.tar.gz
    ```

2) Create ICD file pointing to the new runtime (requires sudo access)

    ```bash
    # OpenCL FPGA emulation RT
    echo  /opt/intel/oclfpgaemu_<fpga_version>/x64/libintelocl_emu.so | sudo tee
      /etc/OpenCL/vendors/intel_fpgaemu.icd
    # OpenCL CPU RT
    echo /opt/intel/oclcpuexp_<cpu_version>/x64/libintelocl.so | sudo tee
      /etc/OpenCL/vendors/intel_expcpu.icd
    ```

3) Extract or build TBB libraries using links in
[the dependency configuration file](../../devops/dependencies.json). For example,
for the archive oneapi-tbb-<tbb_version>-lin.tgz:

    ```bash
    mkdir -p /opt/intel
    cd /opt/intel
    tar -zxvf oneapi-tbb*lin.tgz
    ```

4) Copy files from or create symbolic links to TBB libraries in OpenCL RT
folder:

    ```bash
    # OpenCL FPGA emulation RT
    ln -s /opt/intel/oneapi-tbb-<tbb_version>/lib/intel64/gcc4.8/libtbb.so
      /opt/intel/oclfpgaemu_<fpga_version>/x64/libtbb.so
    ln -s /opt/intel/oneapi-tbb-<tbb_version>/lib/intel64/gcc4.8/libtbbmalloc.so
      /opt/intel/oclfpgaemu_<fpga_version>/x64/libtbbmalloc.so
    ln -s /opt/intel/oneapi-tbb-<tbb_version>/lib/intel64/gcc4.8/libtbb.so.12
      /opt/intel/oclfpgaemu_<fpga_version>/x64/libtbb.so.12
    ln -s /opt/intel/oneapi-tbb-<tbb_version>/lib/intel64/gcc4.8/libtbbmalloc.so.2
      /opt/intel/oclfpgaemu_<fpga_version>/x64/libtbbmalloc.so.2
    # OpenCL CPU RT
    ln -s /opt/intel/oneapi-tbb-<tbb_version>/lib/intel64/gcc4.8/libtbb.so
      /opt/intel/oclcpuexp_<cpu_version>/x64/libtbb.so
    ln -s /opt/intel/oneapi-tbb-<tbb_version>/lib/intel64/gcc4.8/libtbbmalloc.so
      /opt/intel/oclcpuexp_<cpu_version>/x64/libtbbmalloc.so
    ln -s /opt/intel/oneapi-tbb-<tbb_version>/lib/intel64/gcc4.8/libtbb.so.12
      /opt/intel/oclcpuexp_<cpu_version>/x64/libtbb.so.12
    ln -s /opt/intel/oneapi-tbb-<tbb_version>/lib/intel64/gcc4.8/libtbbmalloc.so.2
      /opt/intel/oclcpuexp_<cpu_version>/x64/libtbbmalloc.so.2
    ```

5) Configure library paths (requires sudo access)

    ```bash
    echo /opt/intel/oclfpgaemu_<fpga_version>/x64 | sudo tee
      /etc/ld.so.conf.d/libintelopenclexp.conf
    echo /opt/intel/oclcpuexp_<cpu_version>/x64 | sudo tee -a
      /etc/ld.so.conf.d/libintelopenclexp.conf
    sudo ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
    ```

**Windows (64-bit)**:

1) If you need OpenCL runtime for Intel `GPU` as well, then update/install it
first. Do it **before** installing OpenCL runtime for Intel `CPU` runtime as
OpenCL runtime for Intel `GPU` installer may re-write some important
files or settings and make existing OpenCL runtime for Intel `CPU` runtime
not working properly.

2) Extract the archive with OpenCL runtime for Intel `CPU` and/or for Intel
`FPGA` emulation using links in
[the dependency configuration file](../../devops/dependencies.json).  For
example, to `c:\oclcpu_rt_<cpu_version>`.

3) Extract the archive with TBB runtime or build it from sources using links
in [the dependency configuration file](../../devops/dependencies.json).  For
example, to `c:\oneapi-tbb-<tbb_version>`.

4) Run `Command Prompt` as `Administrator`. To do that click `Start` button,
type `Command Prompt`, click the Right mouse button on it, then click
`Run As Administrator`, then click `Yes` to confirm.

5) In the opened windows run `install.bat` provided with the extracted files
to install runtime to the system and setup environment variables. So, if the
extracted files are in `c:\oclcpu_rt_<cpu_version>\` folder, then type the
command:

    ```bash
    # Install OpenCL FPGA emulation RT
    # Answer Y to clean previous OCL_ICD_FILENAMES configuration and ICD records cleanup
    c:\oclfpga_rt_<fpga_version>\install.bat c:\oneapi-tbb-<tbb_version>\redist\intel64\vc14
    # Install OpenCL CPU RT
    # Answer N for ICD records cleanup
    c:\oclcpu_rt_<cpu_version>\install.bat c:\oneapi-tbb-<tbb_version>\redist\intel64\vc14
    ```

### Obtain prerequisites for ahead of time (AOT) compilation

[Ahead of time compilation](design/CompilerAndRuntimeDesign.md#ahead-of-time-aot-compilation)
requires ahead of time compiler available in `PATH`. There is
AOT compiler for each device type:

* `GPU`, Level Zero and OpenCL runtimes are supported,
* `CPU`, OpenCL runtime is supported,
* `Accelerator` (FPGA or FPGA emulation), OpenCL runtime is supported.

#### GPU

* Linux

  There are two ways how to obtain GPU AOT compiler `ocloc`:
  * (Ubuntu) Download and install intel-ocloc_***.deb package from
    [intel/compute-runtime releases](https://github.com/intel/compute-runtime/releases).
    This package should have the same version as Level Zero / OpenCL GPU
    runtimes installed on the system.
  * (other distros) `ocloc` is a part of
    [Intel&reg; software packages for general purpose GPU capabilities](https://dgpu-docs.intel.com/index.html).

* Windows

  * GPU AOT compiler `ocloc` is a part of
    [Intel&reg; oneAPI Base Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html)
    (Intel&reg; oneAPI DPC++/C++ Compiler component).  
    Make sure that the following path to `ocloc` binary is available in `PATH`
    environment variable:

    * `<oneAPI installation location>/compiler/<version>/windows/lib/ocloc`

#### CPU

* CPU AOT compiler `opencl-aot` is enabled by default. For more, see
[opencl-aot documentation](https://github.com/intel/llvm/blob/sycl/opencl/opencl-aot/README.md).

#### Accelerator

* Accelerator AOT compiler `aoc` is a part of
[Intel&reg; oneAPI Base Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html)
(Intel&reg; oneAPI DPC++/C++ Compiler component).  
Make sure that these binaries are available in `PATH` environment variable:

  * `aoc` from `<oneAPI installation location>/compiler/<version>/<OS>/lib/oclfpga/bin`
  * `aocl-ioc64` from `<oneAPI installation location>/compiler/<version>/<OS>/bin`

### Test DPC++ toolchain

#### Run in-tree LIT tests

To verify that built DPC++ toolchain is working correctly, run:

**Linux**:

```bash
python $DPCPP_HOME/llvm/buildbot/check.py
```

**Windows (64-bit)**:

```bat
python %DPCPP_HOME%\llvm\buildbot\check.py
```

Make sure that psutil package is installed.
If no OpenCL GPU/CPU runtimes are available, the corresponding tests are
skipped.

If CUDA support has been built, it is tested only if there are CUDA devices
available.

If testing with HIP for AMD, the lit tests will use `gfx906` as the default
architecture. It is possible to change it by adding
`-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=<target>` to the CMake
variable `SYCL_CLANG_EXTRA_FLAGS`.

#### Run DPC++ E2E tests

Follow instructions from the link below to build and run tests:
[README](https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/README.md#build-and-run-tests)

#### Run Khronos\* SYCL\* conformance test suite (optional)

Khronos\* SYCL\* conformance test suite (CTS) is intended to validate
implementation conformance to Khronos\* SYCL\* specification. DPC++ compiler is
expected to pass significant number of tests, and it keeps improving.

Follow Khronos\* SYCL\* CTS instructions from
[README](https://github.com/KhronosGroup/SYCL-CTS#configuration--compilation)
file to obtain test sources and instructions how build and execute the tests.

### Run simple DPC++ application

A simple DPC++ or SYCL\* program consists of following parts:

1. Header section
2. Allocating buffer for data
3. Creating SYCL queue
4. Submitting command group to SYCL queue which includes the kernel
5. Wait for the queue to complete the work
6. Use buffer accessor to retrieve the result on the device and verify the data
7. The end

Creating a file `simple-sycl-app.cpp` with the following C++/SYCL code:

```c++
#include <sycl/sycl.hpp>

int main() {
  // Creating buffer of 4 elements to be used inside the kernel code
  sycl::buffer<size_t, 1> Buffer(4);

  // Creating SYCL queue
  sycl::queue Queue;

  // Size of index space for kernel
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler &cgh) {
    // Getting write only access to the buffer on a device.
    sycl::accessor Accessor{Buffer, cgh, sycl::write_only};
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(
        NumOfWorkItems, [=](sycl::id<1> WIid) {
          // Fill buffer with indexes.
          Accessor[WIid] = WIid.get(0);
        });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};

  // Check the results
  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return MismatchFound;
}

```

To build simple-sycl-app put `bin` and `lib` to PATHs:

**Linux**:

```bash
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
```

**Windows (64-bit)**:

```bat
set PATH=%DPCPP_HOME%\llvm\build\bin;%PATH%
set LIB=%DPCPP_HOME%\llvm\build\lib;%LIB%
```

and run following command:

```bash
clang++ -fsycl simple-sycl-app.cpp -o simple-sycl-app.exe
```

When building for CUDA or HIP NVIDIA, use the CUDA target triple as follows:

```bash
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
  simple-sycl-app.cpp -o simple-sycl-app-cuda.exe
```

**Linux & Windows (64-bit)**:

```bash
./simple-sycl-app.exe
The results are correct!
```

**NOTE**: oneAPI DPC++/SYCL developers can specify SYCL device for execution
using device selectors (e.g. `sycl::cpu_selector_v`, `sycl::gpu_selector_v`,
[Intel FPGA selector(s)](extensions/supported/sycl_ext_intel_fpga_device_selector.asciidoc))
as explained in following section
[Code the program for a specific GPU](#code-the-program-for-a-specific-gpu).

#### AOT Target architectures

**NOTE**: When building for HIP AMD, you **MUST** use the AMD target triple and
specify the target architecture with
`-Xsycl-target-backend --offload-arch=<arch>` as follows:

```bash
clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa \
  -Xsycl-target-backend --offload-arch=gfx906              \
  simple-sycl-app.cpp -o simple-sycl-app-amd.exe
```

The target architecture may also be specified for the CUDA backend, with
`-Xsycl-target-backend --cuda-gpu-arch=<arch>`. Specifying the architecture is
necessary if an application aims to use newer hardware features, such as
native atomic operations or the joint_matrix extension.
Moreover, it is possible to pass specific options to CUDA `ptxas` (such as
`--maxrregcount=<n>` for limiting the register usage or `--verbose` for
printing generation statistics) using the `-Xcuda-ptxas` flag.

```bash
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
  simple-sycl-app.cpp -o simple-sycl-app-cuda.exe \
  -Xcuda-ptxas --maxrregcount=128 -Xcuda-ptxas --verbose \
  -Xsycl-target-backend --cuda-gpu-arch=sm_80
```

Additionally AMD and Nvidia targets also support aliases for the target to
simplify passing the specific architectures, for example
`-fsycl-targets=nvidia_gpu_sm_80` is equivalent to
`-fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend
--cuda-gpu-arch=sm_80`, the full list of available aliases is documented in the
 [Users Manual](UsersManual.md#generic-options), for the `-fsycl-targets`
 option.

To build simple-sycl-app ahead of time for GPU, CPU or Accelerator devices,
specify the target architecture.  The examples provided use a supported
alias for the target, representing a full triple.  Additional details can
be found in the [Users Manual](UsersManual.md#generic-options).

```-fsycl-targets=spir64_gen``` for GPU,
```-fsycl-targets=spir64_x86_64``` for CPU,
```-fsycl-targets=spir64_fpga``` for Accelerator.

Multiple target architectures are supported.

E.g., this command builds simple-sycl-app for GPU and CPU devices in
ahead of time mode:

```bash
clang++ -fsycl -fsycl-targets=spir64_gen,spir64_x86_64 simple-sycl-app.cpp -o simple-sycl-app-aot.exe
```

Additionally, user can pass specific options of AOT compiler to
the DPC++ compiler using ```-Xsycl-target-backend``` option, see
[Device code formats](design/CompilerAndRuntimeDesign.md#device-code-formats) for
more. To find available options, execute:

```ocloc compile --help``` for GPU,
```opencl-aot --help``` for CPU,
```aoc -help -sycl``` for Accelerator.

The `simple-sycl-app.exe` application doesn't specify SYCL device for
execution, so SYCL runtime will use `default_selector` logic to select one
of accelerators available in the system.
In this case, the behavior of the `default_selector` can be altered
using the `ONEAPI_DEVICE_SELECTOR` environment variable, setting `cuda:*` forces
the usage of the CUDA backend (if available), `hip:*` forces
the usage of the HIP backend (if available), `opencl:*` will
force the usage of the OpenCL backend.

```bash
ONEAPI_DEVICE_SELECTOR=cuda:* ./simple-sycl-app-cuda.exe
```

The default is the OpenCL backend if available.

**NOTE**: `nvptx64-nvidia-cuda` is usable with `-fsycl-targets`
if clang was built with the cmake option `SYCL_ENABLE_BACKENDS=cuda`.

### Build DPC++ application with CMake

DPC++ applications can be built with CMake by simply using DPC++ as the C++
compiler and by adding the SYCL specific flags. For example assuming `clang++`
is on the `PATH`, a minimal `CMakeLists.txt` file for the sample above would be:

```cmake
# Modifying the compiler should be done before the project line
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

project(simple-sycl-app LANGUAGES CXX)

add_executable(simple-sycl-app simple-sycl-app.cpp)
```

NOTE: compiling SYCL programs requires passing the SYCL flags to `clang++` for
both the compilation and linking stages, so using `add_compile_options` to pass
the SYCL flags is not enough on its own, they should also be passed to
`add_link_options`, or more simply the SYCL flags can just be added to
`CMAKE_CXX_FLAGS`.

NOTE: When linking a SYCL application, `clang++` will implicitly link it against
`libsycl.so`, so there is no need to add `-lsycl` to `target_link_libraries` in
the CMake.

### Code the program for a specific GPU

To assist in finding a specific SYCL compatible device out of all that may be
available, a "device selector" may be used. A "device selector" is a ranking
function (C++ Callable) that will give an integer ranking value to all the
devices on the system. It can be passed to `sycl::queue`, `sycl::device` and
`sycl::platform` constructors. The highest ranking device is then selected. SYCL
has built-in device selectors for selecting a generic GPU, CPU, or accelerator
device, as well as one for a default device. Additionally, a user can define
their own as function, lambda, or functor class. Device selectors returning
negative values will "reject" a device ensuring it is not selected, but values 0
or higher will be selected by the highest score with ties resolved by an
internal algorithm (see Section 4.6.1 of the SYCL 2020 specification)

The example below illustrates how to use a device selector to create device and
queue objects bound to Intel GPU device:

```c++
#include <sycl/sycl.hpp>

int main() {

  auto NEOGPUDeviceSelector = [](const sycl::device &Device){
    using namespace sycl::info;

    const std::string DeviceName = Device.get_info<device::name>();
    bool match = Device.is_gpu() && (DeviceName.find("HD Graphics NEO") != std::string::npos);
    return match ? 1 : -1;
  };

  try {
    sycl::queue Queue(NEOGPUDeviceSelector);
    sycl::device Device(NEOGPUDeviceSelector);
  } catch (sycl::exception &E) {
    std::cout << E.what() << std::endl;
  }
}

```

The device selector below selects an NVIDIA device only, and won't execute if
there is none.

```c++

int CUDASelector(const sycl::device &Device) {
  using namespace sycl::info;
  const std::string DriverVersion = Device.get_info<device::driver_version>();

  if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
    std::cout << " CUDA device found " << std::endl;
    return 1;
  };
  return -1;
}

```

## C++ standard

* DPC++ runtime and headers require C++17 at least.
* DPC++ compiler builds apps as C++17 apps by default. Higher versions of
  standard are supported as well.

## Known Issues and Limitations

* SYCL 2020 support work is in progress.
* 32-bit host/target is not supported.
* DPC++ works only with OpenCL low level runtimes which support out-of-order
  queues.
* On Windows linking DPC++ applications with `/MTd` flag is known to cause
  crashes.

### CUDA back-end limitations

* Windows support is currently experimental and not regularly tested.
* `sycl::sqrt` is not correctly rounded by default as the SYCL specification
  allows lower precision, when porting from CUDA it may be helpful to use
  `-fsycl-fp32-prec-sqrt` to use the correctly rounded square root, this is
  significantly slower but matches the default precision used by `nvcc`, and
  this `clang++` flag is equivalent to the `nvcc` `-prec-sqrt` flag, except that
  it defaults to `false`.
* No Opt (O0) uses the IPSCCP compiler pass by default, although the IPSCCP pass
  can be switched off at O0 using the `-mllvm -use-ipsccp-nvptx-O0=false` flag at
  the user's discretion.
  The reason that the IPSCCP pass is used by default even at O0 is that there is
  currently an unresolved issue with the nvvm-reflect compiler pass: This pass is
  used to pick the correct branches depending on the SM version which can be
  optionally specified by the `--cuda-gpu-arch` flag.
  If the arch flag is not specified by the user, the default value, SM 50, is used.
  Without the execution of the IPSCCP pass at -O0 when using a low SM version,
  dead instructions which require a higher SM version can remain. Since
  corresponding issues occur in other backends future work will aim for a
  universal solution to these issues.

### HIP back-end limitations

* Requires a ROCm compatible system and GPU, see for
  [Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-skus)
  and for
  [Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html#supported-skus).
* Windows for HIP is not supported by DPC++ at the moment so it may not work.
* `printf` within kernels is not supported.
* C++ standard library functions using complex types are not supported,
  `sycl::complex` should be used instead.

## Find More

* [oneAPI specifications](https://spec.oneapi.io/versions/latest/)
* [SYCL\* specification](https://www.khronos.org/registry/SYCL)
* [Level Zero specification](https://spec.oneapi.io/level-zero/latest/index.html)

<sub>\*Other names and brands may be claimed as the property of others.</sub>
