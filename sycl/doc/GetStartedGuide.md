# Overview

The DPC++ Compiler compiles C++ and SYCL\* source files with code for both CPU
and a wide range of compute accelerators such as GPU and FPGA.

# Table of contents

* [Prerequisites](#prerequisites)
  * [Create DPC++ workspace](#create-dpc-workspace)
* [Build DPC++ toolchain](#build-dpc-toolchain)
  * [Build DPC++ toolchain with libc++ library](#build-dpc-toolchain-with-libc-library)
  * [Build DPC++ toolchain with support for NVIDIA CUDA](#build-dpc-toolchain-with-support-for-nvidia-cuda)
* [Use DPC++ toolchain](#use-dpc-toolchain)
  * [Install low level runtime](#install-low-level-runtime)
  * [Test DPC++ toolchain](#test-dpc-toolchain)
  * [Run simple DPC++ application](#run-simple-dpc-application)
* [C++ standard](#c-standard)
* [Known Issues and Limitations](#known-issues-and-limitations)
* [CUDA backend limitations](#cuda-backend-limitations)
* [Find More](#find-more)

# Prerequisites

* `git` - https://git-scm.com/downloads
* `cmake` version 3.2 or later - http://www.cmake.org/download/
* `python` - https://www.python.org/downloads/release/python-2716/
* C++ compiler
  * Linux: `GCC` version 5.1.0 or later (including libstdc++) -
    https://gcc.gnu.org/install/
  * Windows: `Visual Studio` version 15.7 preview 4 or later -
    https://visualstudio.microsoft.com/downloads/

## Create DPC++ workspace

Throughout this document `DPCPP_HOME` denotes the path to the local directory
created as DPC++ workspace. It might be useful to create an environment variable
with the same name.

**Linux**

```bash
export DPCPP_HOME=/export/home/sycl_workspace
mkdir $DPCPP_HOME
```

**Windows (64-bit)**

Open a developer command prompt using one of two methods:

- Click start menu and search for "**x64** Native Tools Command Prompt for VS
  XXXX", where XXXX is a version of installed Visual Studio.
- Ctrl-R, write "cmd", click enter, then run
  `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64`

```bat
set DPCPP_HOME=%USERPROFILE%\sycl_workspace
mkdir %DPCPP_HOME%
```

# Build DPC++ toolchain

**Linux**
```bash
cd $DPCPP_HOME
git clone https://github.com/intel/llvm -b sycl
mkdir $DPCPP_HOME/build
cd $DPCPP_HOME/build

cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" \
-DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl" \
-DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl" \
-DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$DPCPP_HOME/llvm/sycl \
-DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$DPCPP_HOME/llvm/llvm-spirv \
$DPCPP_HOME/llvm/llvm

make -j`nproc` sycl-toolchain
```

**Windows (64-bit)**
```bat
cd %DPCPP_HOME%
git clone https://github.com/intel/llvm -b sycl
mkdir %DPCPP_HOME%\build
cd %DPCPP_HOME%\build

cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" ^
-DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl" ^
-DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl" ^
-DLLVM_EXTERNAL_SYCL_SOURCE_DIR="%DPCPP_HOME%\llvm\sycl" ^
-DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR="%DPCPP_HOME%\llvm\llvm-spirv" ^
-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
"%DPCPP_HOME%\llvm\llvm"

ninja sycl-toolchain
```

To use ahead-of-time compilation for the Intel&reg; processors, additionally
build opencl-aot target:

1. add ```opencl-aot``` to ```-DLLVM_EXTERNAL_PROJECTS``` and
```-DLLVM_ENABLE_PROJECTS``` variables above
2. add ```opencl-aot``` to
```make``` (for Linux) or ```ninja``` (for Windows) commands above

For more, see [opencl-aot documentation](../../opencl-aot/README.md).

TODO: add instructions how to deploy built DPC++ toolchain.

## Build DPC++ toolchain with libc++ library

There is experimental support for building and linking DPC++ runtime with
libc++ library instead of libstdc++. To enable it the following CMake options
should be used.

**Linux**
```
-DSYCL_USE_LIBCXX=ON \
-DSYCL_LIBCXX_INCLUDE_PATH=<path to libc++ headers> \
-DSYCL_LIBCXX_LIBRARY_PATH=<path to libc++ and libc++abi libraries>
```

## Build DPC++ toolchain with support for NVIDIA CUDA

There is experimental support for DPC++ for CUDA devices.

To enable support for CUDA devices, the following arguments need to be added to
the CMake command when building the DPC++ compiler.

```
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
-DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl;libclc" \
-DSYCL_BUILD_PI_CUDA=ON \
-DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
-DLIBCLC_TARGETS_TO_BUILD="nvptx64--;nvptx64--nvidiacl"
```

Enabling this flag requires an installation of
[CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2) on the system,
refer to
[NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Currently, the only combination tested is Ubuntu 18.04 with CUDA 10.2 using
a Titan RTX GPU (SM 71), but it should work on any GPU compatible with SM 50 or
above.

# Use DPC++ toolchain

## Install low level runtime

To run DPC++ applications on OpenCL devices, OpenCL implementation(s) must be
present in the system.

Please, refer to [the Release Notes](../ReleaseNotes.md) for recommended Intel
runtime versions.

The `GPU` runtime that is needed to run DPC++ application on Intel `GPU` devices
can be downloaded from the following web pages:

* Linux: [Intel&reg; Graphics Compute Runtime for
   OpenCL&trade;](https://github.com/intel/compute-runtime/releases)

* Windows: [Intel&reg; Download
   Center](https://downloadcenter.intel.com/product/80939/Graphics-Drivers)


To install Intel `CPU` runtime for OpenCL devices the corresponding runtime
asset/archive should be downloaded from
[DPC++ Compiler and Runtime updates](../ReleaseNotes.md) and installed following
procedure below.

Intel `CPU` runtime for OpenCL depends on Threading Building Blocks library
which should be downloaded from [Threading Building Blocks (TBB)
 GitHub repository](https://github.com/intel/tbb) and installed following
procedure below.

Intel `CPU` runtime for OpenCL devices can be switched into Intel FPGA
Emulation device for OpenCL. The following parameter should be set in `cl.cfg`
file (available in directory containing CPU runtime for OpenCL) to switch
OpenCL device mode:
```
CL_CONFIG_DEVICES = fpga-emu
```

**Linux**

1) Extract the archive. For example, for the archive
`oclcpu_rt_<cpu_version>.tar.gz` you would run the following commands
```bash
mkdir -p /opt/intel/oclcpuexp_<cpu_version>
cd /opt/intel/oclcpuexp_<cpu_version>
tar -zxvf oclcpu_rt_<cpu_version>.tar.gz
```
2) Create ICD file pointing to the new runtime
```bash
echo /opt/intel/oclcpuexp_<cpu_version>/x64/libintelocl.so >
  /etc/OpenCL/vendors/intel_expcpu.icd
```

3) Extract TBB libraries. For example, for the archive tbb-<tbb_version>-lin.tgz

```bash
mkdir -p /opt/intel/tbb_<tbb_version>
cd /opt/intel/tbb_<tbb_version>
tar -zxvf tbb*lin.tgz
```

4) Copy files from or create symbolic links to TBB libraries in OpenCL RT
folder:
```bash
ln -s /opt/intel/tbb_<tbb_version>/tbb/lib/intel64/gcc4.8/libtbb.so
  /opt/intel/oclcpuexp/x64/libtbb.so
ln -s /opt/intel/tbb_<tbb_version>/tbb/lib/intel64/gcc4.8/libtbbmalloc.so
  /opt/intel/oclcpuexp/x64/libtbbmalloc.so
ln -s /opt/intel/tbb_<tbb_version>/tbb/lib/intel64/gcc4.8/libtbb.so.2
  /opt/intel/oclcpuexp/x64/libtbb.so.2
ln -s /opt/intel/tbb_<tbb_version>/tbb/lib/intel64/gcc4.8/libtbbmalloc.so.2
  /opt/intel/oclcpuexp/x64/libtbbmalloc.so.2
```

5) Configure library paths
```bash
echo /opt/intel/oclcpuexp_<cpu_version>/x64 >
  /etc/ld.so.conf.d/libintelopenclexp.conf
ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
```
**Windows (64-bit)**
1) If you need `GPU` as well, then update/install it first. Do it **before**
installing `CPU` runtime as `GPU` runtime installer may re-write some important
files or settings and make existing `CPU` runtime not working properly.

2) Extract the archive to some folder. For example, to
`c:\oclcpu_rt_<cpu_version>` and `c:\tbb_<tbb_version>`.

3) Run `Command Prompt` as `Administrator`. To do that click `Start` button,
type `Command Prompt`, click the Right mouse button on it, then click
`Run As Administrator`, then click `Yes` to confirm.

4) In the opened windows run `install.bat` provided with the extracted files
to install runtime to the system and setup environment variables. So, if the
extracted files are in `c:\oclcpu_rt_<cpu_version>\` folder, then type the
command:
```bash
c:\oclcpu_rt_<cpu_version>\install.bat c:\tbb_<tbb_version>\tbb\bin\intel64\vc14
```

## Test DPC++ toolchain

### Run regression tests

To verify that built DPC++ toolchain is working correctly, run:

**Linux**
```bash
make -j`nproc` check-all
```

**Windows (64-bit)**
```bat
ninja check-all
```

If no OpenCL GPU/CPU runtimes are available, the corresponding tests are
skipped.

### Run Khronos\* SYCL\* conformance test suite (optional)

Khronos\* SYCL\* conformance test suite (CTS) is intended to validate
implementation conformance to Khronos\* SYCL\* specification. DPC++ compiler is
expected to pass significant number of tests, and it keeps improving.

Follow Khronos\* SYCL\* CTS instructions from
[README](https://github.com/KhronosGroup/SYCL-CTS#sycl-121-conformance-test-suite)
file to obtain test sources and instructions how build and execute the tests.

To configure testing of DPC++ toochain set
`SYCL_IMPLEMENTATION=Intel_SYCL` and
`Intel_SYCL_ROOT=<path to the SYCL installation>` CMake variables.

**Linux**
```bash
cmake -DIntel_SYCL_ROOT=$DPCPP_HOME/deploy -DSYCL_IMPLEMENTATION=Intel_SYCL ...
```

**Windows (64-bit)**
```bat
cmake -DIntel_SYCL_ROOT=%DPCPP_HOME%\deploy -DSYCL_IMPLEMENTATION=Intel_SYCL ...
```

### Build Doxygen documentation

Building Doxygen documentation is similar to building the product itself. First,
the following tools need to be installed:
- doxygen
- graphviz

Then you'll need to add the following options to your CMake configuration
command:
```
-DLLVM_ENABLE_DOXYGEN=ON
```

After CMake cache is generated, build the documentation with `doxygen-sycl`
target. It will be put to `/path/to/build/tools/sycl/doc/html` directory.

## Run simple DPC++ application

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
#include <CL/sycl.hpp>

int main() {
  // Creating buffer of 4 ints to be used inside the kernel code
  cl::sycl::buffer<cl::sycl::cl_int, 1> Buffer(4);

  // Creating SYCL queue
  cl::sycl::queue Queue;

  // Size of index space for kernel
  cl::sycl::range<1> NumOfWorkItems{Buffer.get_count()};

  // Submitting command group(work) to queue
  Queue.submit([&](cl::sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<cl::sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(
        NumOfWorkItems, [=](cl::sycl::id<1> WIid) {
          // Fill buffer with indexes
          Accessor[WIid] = (cl::sycl::cl_int)WIid.get(0);
        });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  const auto HostAccessor = Buffer.get_access<cl::sycl::access::mode::read>();

  // Check the results
  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.get_count(); ++I) {
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

**Linux**
```bash
export PATH=$DPCPP_HOME/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/build/lib:$LD_LIBRARY_PATH
```

**Windows (64-bit)**
```bat
set PATH=%DPCPP_HOME%\build\bin;%PATH%
set LIB=%DPCPP_HOME%\build\lib;%LIB%
```

and run following command:

```bash
clang++ -fsycl simple-sycl-app.cpp -o simple-sycl-app.exe
```

When building for CUDA, use the CUDA target triple as follows:

```bash
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
  simple-sycl-app.cpp -o simple-sycl-app-cuda.exe
```

This `simple-sycl-app.exe` application doesn't specify SYCL device for
execution, so SYCL runtime will use `default_selector` logic to select one
of accelerators available in the system or SYCL host device.

Note: `nvptx64-nvidia-cuda-sycldevice` is usable with `-fsycl-targets`
if clang was built with the cmake option `SYCL_BUILD_PI_CUDA=ON`.

**Linux & Windows**
```bash
./simple-sycl-app.exe
The results are correct!
```
**Note**:
Currently, when the application has been built with the CUDA target, the CUDA
backend must be selected at runtime using the `SYCL_BE` environment variable.

```bash
SYCL_BE=PI_CUDA ./simple-sycl-app-cuda.exe
```

NOTE: DPC++/SYCL developer can specify SYCL device for execution using device
selectors (e.g. `cl::sycl::cpu_selector`, `cl::sycl::gpu_selector`,
[Intel FPGA selector(s)](extensions/IntelFPGA/FPGASelector.md)) as
explained in following section [Code the program for a specific
GPU](#code-the-program-for-a-specific-gpu).

## Code the program for a specific GPU

To specify OpenCL device SYCL provides the abstract `cl::sycl::device_selector`
class which the can be used to define how the runtime should select the best
device.

The method `cl::sycl::device_selector::operator()` of the SYCL
`cl::sycl::device_selector` is an abstract member function which takes a
reference to a SYCL device and returns an integer score. This abstract member
function can be implemented in a derived class to provide a logic for selecting
a SYCL device. SYCL runtime uses the device for with the highest score is
returned. Such object can be passed to `cl::sycl::queue` and `cl::sycl::device`
constructors.

The example below illustrates how to use `cl::sycl::device_selector` to create
device and queue objects bound to Intel GPU device:

```c++
#include <CL/sycl.hpp>

int main() {
  class NEOGPUDeviceSelector : public cl::sycl::device_selector {
  public:
    int operator()(const cl::sycl::device &Device) const override {
      using namespace cl::sycl::info;

      const std::string DeviceName = Device.get_info<device::name>();
      const std::string DeviceVendor = Device.get_info<device::vendor>();

      return Device.is_gpu() && (DeviceName.find("HD Graphics NEO") != std::string::npos);
    }
  };

  NEOGPUDeviceSelector Selector;
  try {
    cl::sycl::queue Queue(Selector);
    cl::sycl::device Device(Selector);
  } catch (cl::sycl::invalid_parameter_error &E) {
    std::cout << E.what() << std::endl;
  }
}

```

The device selector below selects an NVIDIA device only, and won't
execute if there is none.

```c++
class CUDASelector : public cl::sycl::device_selector {
  public:
    int operator()(const cl::sycl::device &Device) const override {
      using namespace cl::sycl::info;

      const std::string DeviceName = Device.get_info<device::name>();
      const std::string DeviceVendor = Device.get_info<device::vendor>();

      if (Device.is_gpu() && (DeviceName.find("NVIDIA") != std::string::npos)) {
        return 1;
      };
      return -1;
    }
};
```

# C++ standard

- Minimal supported C++ standard is C++11 on Linux and C++14 on Windows.

# Known Issues and Limitations

- DPC++ device compiler fails if the same kernel was used in different
  translation units.
- SYCL host device is not fully supported.
- 32-bit host/target is not supported.
- DPC++ works only with OpenCL low level runtimes which support out-of-order
  queues.
- On Windows linking DPC++ applications with `/MTd` flag is known to cause
  crashes.

## CUDA back-end limitations

- Backend is only supported on Linux
- The only combination tested is Ubuntu 18.04 with CUDA 10.2 using a Titan RTX
  GPU (SM 71), but it should work on any GPU compatible with SM 50 or above
- The NVIDIA OpenCL headers conflict with the OpenCL headers required for this
  project and may cause compilation issues on some platforms

# Find More

DPC++ specification:
[https://spec.oneapi.com/versions/latest/elements/dpcpp/source/index.html](https://spec.oneapi.com/versions/latest/elements/dpcpp/source/index.html)
SYCL\* 1.2.1 specification:
[www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf)


\*Other names and brands may be claimed as the property of others.

