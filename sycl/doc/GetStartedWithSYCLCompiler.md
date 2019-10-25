# Overview

The SYCL* Compiler compiles C++\-based SYCL source files with code for both CPU
and a wide range of compute accelerators. The compiler uses Khronos*
OpenCL&trade; API to offload computations to accelerators.

# Table of contents

* [Prerequisites](#prerequisites)
  * [Create SYCL workspace](#create-sycl-workspace)
* [Build SYCL toolchain](#build-sycl-toolchain)
  * [Build SYCL toolchain with libc++ library](#build-sycl-toolchain-with-libc-library)
* [Use SYCL toolchain](#use-sycl-toolchain)
  * [Install low level runtime](#install-low-level-runtime)
  * [Test SYCL toolchain](#test-sycl-toolchain)
  * [Run simple SYCL application](#run-simple-sycl-application)
* [C++ standard](#c-standard)
* [Known Issues and Limitations](#known-issues-and-limitations)
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

## Create SYCL workspace

Throughout this document `SYCL_HOME` denotes the path to the local directory
created as SYCL workspace. It might be useful to create an environment variable
with the same name.

**Linux**

```bash
export SYCL_HOME=/export/home/sycl_workspace
mkdir $SYCL_HOME
```

**Windows (64-bit)**

Open a developer command prompt using one of two methods:

- Click start menu and search for "**x64** Native Tools Command Prompt for VS XXXX", where
  XXXX is a version of installed Visual Studio.
- Ctrl-R, write "cmd", click enter, then run
  `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64`

```bat
set SYCL_HOME=%USERPROFILE%\sycl_workspace
mkdir %SYCL_HOME%
```

# Build SYCL toolchain

**Linux**
```bash
cd $SYCL_HOME
git clone https://github.com/intel/llvm -b sycl
mkdir $SYCL_HOME/build
cd $SYCL_HOME/build

cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" \
-DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl" \
-DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl" \
-DLLVM_EXTERNAL_SYCL_SOURCE_DIR=$SYCL_HOME/llvm/sycl \
-DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=$SYCL_HOME/llvm/llvm-spirv \
$SYCL_HOME/llvm/llvm

make -j`nproc` sycl-toolchain
```

**Windows (64-bit)**
```bat
cd %SYCL_HOME%
git clone https://github.com/intel/llvm -b sycl
mkdir %SYCL_HOME%\build
cd %SYCL_HOME%\build

cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" ^
-DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl" ^
-DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl" ^
-DLLVM_EXTERNAL_SYCL_SOURCE_DIR="%SYCL_HOME%\llvm\sycl" ^
-DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR="%SYCL_HOME%\llvm\llvm-spirv" ^
-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_C_FLAGS="/GS" ^
-DCMAKE_CXX_FLAGS="/GS" -DCMAKE_EXE_LINKER_FLAGS="/NXCompat /DynamicBase" ^
-DCMAKE_SHARED_LINKER_FLAGS="/NXCompat /DynamicBase" ^
"%SYCL_HOME%\llvm\llvm"

ninja sycl-toolchain
```

TODO: add instructions how to deploy built SYCL toolchain.

## Build SYCL toolchain with libc++ library

There is experimental support for building and linking SYCL runtime with
libc++ library instead of libstdc++. To enable it the following CMake options
should be used.

**Linux**
```
-DSYCL_USE_LIBCXX=ON \
-DSYCL_LIBCXX_INCLUDE_PATH=<path to libc++ headers> \
-DSYCL_LIBCXX_LIBRARY_PATH=<path to libc++ and libc++abi libraries>
```

# Use SYCL toolchain

## Install low level runtime

To run SYCL  applications on OpenCL devices, OpenCL implementation(s) must be
present in the system.

Please, refer to [the Release Notes](../ReleaseNotes.md) for recommended Intel
runtime versions.

The `GPU` runtime that is needed to run SYCL application on Intel `GPU` devices
can be downloaded from the following web pages:

* Linux: [Intel&reg; Graphics Compute Runtime for
   OpenCL&trade;](https://github.com/intel/compute-runtime/releases)

* Windows: [Intel&reg; Download
   Center](https://downloadcenter.intel.com/product/80939/Graphics-Drivers)


To install Intel `CPU` runtime for OpenCL devices the corresponding runtime
asset/archive should be downloaded from
[SYCL Compiler and Runtime updates](../ReleaseNotes.md) and installed using
the following procedure.

**Linux**

1) Extract the archive. For example, for the archive
`oclcpu_rt_<new_version>.tar.gz` you would run the following commands
```bash
mkdir -p /opt/intel/oclcpuexp
cd /opt/intel/oclcpuexp
tar -zxvf oclcpu_rt_<new_version>.tar.gz
```
2) Create ICD file pointing to the new runtime
```bash
echo /opt/intel/oclcpuexp/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd
```
3) Configure library paths
```bash
echo /opt/intel/oclcpuexp/x64 > /etc/ld.so.conf.d/libintelopenclexp.conf
ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf
```
**Windows (64-bit)**
1) If you need `GPU` as well, then update/install it first. Do it **before**
installing `CPU` runtime as `GPU` runtime installer may re-write some important
files or settings and make existing `CPU` runtime not working properly.

2) Extract the archive to some folder. For example, to `c:\oclcpu_rt_<new_version>`.

3) Run `Command Prompt` as `Administrator`. To do that click `Start` button,
type `Command Prompt`, click the Right mouse button on it, then click
`Run As Administrator`, then click `Yes` to confirm.

4) In the opened windows run `install.bat` provided with the extracted files
to install runtime to the system and setup environment variables. So, if the
extracted files are in `c:\oclcpu_rt_<new_version>\` folder, then type the
command: `c:\oclcpu_rt_<new_version>\install.bat`

## Test SYCL toolchain

### Run regression tests

To verify that built SYCL toolchain is working correctly, run:

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

### Run Khronos SYCL conformance test suite (optional)

Khronos SYCL conformance test suite (CTS) is intended to validate SYCL
implementation conformance to Khronos SYCL specification.

Follow Khronos SYCL-CTS instructions from
[README](https://github.com/KhronosGroup/SYCL-CTS#sycl-121-conformance-test-suite)
file to obtain test sources and instructions how build and execute the tests.

To configure testing of "Intel SYCL" toochain set
`SYCL_IMPLEMENTATION=Intel_SYCL` and
`Intel_SYCL_ROOT=<path to the SYCL installation>` CMake variables.

**Linux**
```bash
cmake -DIntel_SYCL_ROOT=$SYCL_HOME/deploy -DSYCL_IMPLEMENTATION=Intel_SYCL ...
```

**Windows (64-bit)**
```bat
cmake -DIntel_SYCL_ROOT=%SYCL_HOME%\deploy -DSYCL_IMPLEMENTATION=Intel_SYCL ...
```

## Run simple SYCL application

A simple SYCL program consists of following parts:
1. Header section
2. Allocating buffer for data
3. Creating SYCL queue
4. Submitting command group to SYCL queue which includes the kernel
5. Wait for the queue to complete the work
6. Use buffer accessor to retrieve the result on the device and verify the data
7. The end

Creating a file `simple-sycl-app.cpp` with the following C++ SYCL code in it:

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
export PATH=$SYCL_HOME/build/bin:$PATH
export LD_LIBRARY_PATH=$SYCL_HOME/build/lib:$LD_LIBRARY_PATH
```

**Windows (64-bit)**
```bat
set PATH=%SYCL_HOME%\build\bin;%PATH%
set LIB=%SYCL_HOME%\build\lib;%LIB%
```

and run following command:

```bash
clang++ -fsycl simple-sycl-app.cpp -o simple-sycl-app.exe
```

This `simple-sycl-app.exe` application doesn't specify SYCL device for
execution, so SYCL runtime will use `default_selector` logic to select one
of accelerators available in the system or SYCL host device.

**Linux & Windows**
```bash
./simple-sycl-app.exe
The results are correct!
```

NOTE: SYCL developer can specify SYCL device for execution using device
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

# C++ standard
- Minimally support C++ standard is c++11 on Linux and c++14 on Windows.

# Known Issues and Limitations

- SYCL device compiler fails if the same kernel was used in different
  translation units.
- SYCL host device is not fully supported.
- 32-bit host/target is not supported.
- SYCL works only with OpenCL implementations supporting out-of-order queues.
- On Windows linking SYCL applications with `/MTd` flag is known to cause crashes.

# Find More

SYCL 1.2.1 specification: [www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf)
