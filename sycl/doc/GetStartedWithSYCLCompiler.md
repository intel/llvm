# Overview
The SYCL* Compiler compiles C++\-based SYCL source files with code for both CPU
and a wide range of compute accelerators. The compiler uses Khronos*
OpenCL&trade; API to offload computations to accelerators.


# Before You Begin

### Get `OpenCL runtime` for CPU and/or GPU on `Linux`:

   a. OpenCL&trade; runtime for GPU: follow instructions on
[github.com/intel/compute-runtime/releases](https://github.com/intel/compute-runtime/releases)
to install.

   b. Experimental Intel&reg; CPU Runtime for OpenCL&trade; Applications with
SYCL support: follow the instructions under
[SYCL* Compiler and Runtimes](https://github.com/intel/llvm/releases/tag/2019-07)

### Get `OpenCL runtime` for CPU and/or GPU on `Windows`:
   a. OpenCL&trade; runtime for GPU and CPU: download it from
   [Intel&reg; Download Center](https://downloadcenter.intel.com/download/28991/Intel-Graphics-Windows-10-DCH-Drivers)

   b. The latest/experimental Intel&reg; CPU Runtime for OpenCL&trade; Applications with SYCL support
   on Windows `will soon be published` together with Linux runtime at [SYCL* Compiler and Runtimes](https://github.com/intel/llvm/releases)

### Get the required tools:

   a. `git` - for downloading the sources (Get it at https://git-scm.com/downloads)

   b. `cmake` - for building the compiler and tools, version 3.2 or later (Get it at: http://www.cmake.org/download)

   c. `python` - for building the compiler and running tests (Get it at: https://www.python.org/downloads/release/python-2716/ )

   d. `Visual Studio 2017 or later` (Windows only. Get it at: https://visualstudio.microsoft.com/downloads/)


# Configure environment:
For simplicity it is assumed below that the environment variable SYCL_HOME contains path to a folder where the SYCL compiler and runtime will be stored.
### Linux:
```bash
export SYCL_HOME=/export/home/workspaces/sycl_workspace
mkdir $SYCL_HOME
```
### Windows:
Open a developer command prompt using one of tho methods:
- Click start menu and search for the command prompt. So, for MSVC-2017 it is '`x64 Native Tools Command Prompt for VS 2017`'
- run 'cmd' and then '`"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64`'

```bash
set SYCL_HOME=%USERPROFILE%\workspaces\sycl_workspace
mkdir %SYCL_HOME%
```
# Get `OpenCL-Headers` and `OpenCL-ICD-Loader`

**These 2 steps are optional.** The Compiler build process is going to look for available OpenCL SDK on the local machine.
If it finds suitable OpenCL, it will reuse it. Otherwise, it will automatically download `OpenCL-Headers` and `OpenCL-ICD-Loader` from GitHub and build it.
You may want to run these steps if have some unexpected problems caused by `OpenCL-Headers` or `OpenCL-ICD-Loader` at the compiler build phase.

## Get `OpenCL-Headers`
### Linux:
```bash
cd $SYCL_HOME
git clone https://github.com/KhronosGroup/OpenCL-Headers
export OPENCL_HEADERS=$SYCL_HOME/OpenCL-Headers
```

 ### Windows:
```bash
cd %SYCL_HOME%
git clone https://github.com/KhronosGroup/OpenCL-Headers
set OPENCL_HEADERS=%SYCL_HOME%\OpenCL-Headers
```

## Get `OpenCL-ICD-Loader`
You can also find the most recent instructions for this component at https://github.com/KhronosGroup/OpenCL-ICD-Loader

### Linux:
```bash
cd $SYCL_HOME
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
cd OpenCL-ICD-Loader
mkdir build
cd build
cmake -DOPENCL_ICD_LOADER_HEADERS_DIR=$OPENCL_HEADERS ..
make
export ICD_LIB=$SYCL_HOME/OpenCL-ICD-Loader/build/libOpenCL.so
```

### Windows:
```bash
cd %SYCL_HOME%
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
cd OpenCL-ICD-Loader
mkdir build
cd build
cmake -G "Ninja" -DOPENCL_ICD_LOADER_HEADERS_DIR=%OPENCL_HEADERS% -DOPENCL_ICD_LOADER_REQUIRE_WDK=OFF ..
ninja
set ICD_LIB=%SYCL_HOME%\OpenCL-ICD-Loader\build\OpenCL.lib
```

# Checkout and Build the SYCL compiler and runtime

Defining paths to `OpenCL-Headers` and `OpenCL-ICD-Loader` is optional ( `-DOpenCL_INCLUDE_DIR=` and `-DOpenCL_LIBRARY=` ).
If you do not specify the paths explicitly, then:
- If `OpenCL-Headers` and `OpenCL-ICD-Loader` are availalbe in the system, they will be used;
- If they are not available, then OpenCL files are automatically downloaded/built from GitHub during compiler build.

### Linux:
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
-DOpenCL_INCLUDE_DIR="$OPENCL_HEADERS" -DOpenCL_LIBRARY="$ICD_LIB" \
$SYCL_HOME/llvm/llvm

make -j`nproc` sycl-toolchain
```
### Windows:
```bash
mkdir %SYCL_HOME%
cd %SYCL_HOME%
git clone https://github.com/intel/llvm -b sycl
mkdir %SYCL_HOME%\build
cd %SYCL_HOME%\build

cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl" -DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl" -DLLVM_EXTERNAL_SYCL_SOURCE_DIR="%SYCL_HOME%\llvm\sycl" -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR="%SYCL_HOME%\llvm\llvm-spirv" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_C_FLAGS="/GS" -DCMAKE_CXX_FLAGS="/GS" -DCMAKE_EXE_LINKER_FLAGS="/NXCompat /DynamicBase" -DCMAKE_SHARED_LINKER_FLAGS="/NXCompat /DynamicBase" -DOpenCL_INCLUDE_DIR="%OPENCL_HEADERS%" -DOpenCL_LIBRARY="%ICD_LIB%" "%SYCL_HOME%\llvm\llvm"

ninja sycl-toolchain
```

After the build completed, the SYCL compiler/include/libraries can be found
in `$SYCL_HOME/build` directory.

---

## Build the SYCL runtime with libc++ library.

There is experimental support for building and linking SYCL runtime with
libc++ library instead of libstdc++. To enable it the following cmake options
should be used.
### Linux:
```
-DSYCL_USE_LIBCXX=ON \
-DSYCL_LIBCXX_INCLUDE_PATH=<path to libc++ headers> \
-DSYCL_LIBCXX_LIBRARY_PATH=<path to libc++ and libc++abi libraries>
```
# Test the SYCL compiler and runtime

After building the SYCL compiler and runtime, choose the amount of LIT testing you need and run one of the LIT tests suites shown below:
### Linux:
```bash
make -j`nproc` check-all        # to run all test suites including those smaller ones shown below
make -j`nproc` check-llvm       # run llvm tests only
make -j`nproc` check-llvm-spirv # run llvm-spirv tests only
make -j`nproc` check-clang      # run clang tests only
make -j`nproc` check-sycl       # run sycl tests only
```
### Windows:
```bash
ninja check-all
ninja check-llvm
ninja check-llvm-spirv
ninja check-clang
ninja check-sycl
```
If no OpenCL GPU/CPU runtimes are available, the corresponding LIT tests are skipped

# Create a simple SYCL program

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

# Build and Test a simple SYCL program

To build simple-sycl-app put `bin` and `lib` to PATHs and run following command:
### Linux:
   ```bash
   export PATH=$SYCL_HOME/build/bin:$PATH
   export LD_LIBRARY_PATH=$SYCL_HOME/build/lib:$LD_LIBRARY_PATH
   ```
### Windows:
   ```bash
   set PATH=%SYCL_HOME%\build\bin;%PATH%
   set LIB=%SYCL_HOME%\build\lib;%LIB%
   ```

### Linux & Windows:
   ```bash
   clang++ -fsycl simple-sycl-app.cpp -o simple-sycl-app.exe -lOpenCL
   ```

This `simple-sycl-app.exe` application doesn't specify SYCL device for execution,
so SYCL runtime will first try to execute on OpenCL GPU device first, if OpenCL
GPU device is not found, it will try to run OpenCL CPU device; and if OpenCL
CPU device is also not available, SYCL runtime will run on SYCL host device.

### Linux & Windows:
   ```bash
   ./simple-sycl-app.exe
   The results are correct!
   ```


NOTE: SYCL developer can specify SYCL device for execution using device
selectors (e.g. `cl::sycl::cpu_selector`, `cl::sycl::gpu_selector`) as
explained in following section [Code the program for a specific
GPU](#code-the-program-for-a-specific-gpu).

# Code the program for a specific GPU

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

      return Device.is_gpu() && DeviceName.find("HD Graphics NEO") ? 1 : -1;
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

# Known Issues or Limitations

- SYCL device compiler fails if the same kernel was used in different
  translation units.
- SYCL host device is not fully supported.
- SYCL works only with OpenCL implementations supporting out-of-order queues.

# Find More

SYCL 1.2.1 specification: [www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf)
