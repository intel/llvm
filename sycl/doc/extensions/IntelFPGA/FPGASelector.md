# FPGA selector

Intel FPGA users can use header file: `#include<CL/sycl/intel/fpga_device_selector.hpp>` to simplify their code
when they want to specify FPGA hardware device or FPGA emulation device.

## Implementation

Current implementation is based on platform name. This is useful in the most common case when user have
one FPGA board installed in their system (one device per platform). 

## Usage: select FPGA hardware device
```c++
#include <CL/sycl/intel/fpga_device_selector.hpp>
...
// force FPGA hardware device
cl::sycl::queue deviceQueue{cl::sycl::intel::fpga_selector{}};
...
```

## Usage: select FPGA emulator device
```c++
#include <CL/sycl/intel/fpga_device_selector.hpp>
...
// force FPGA emulation device
cl::sycl::queue deviceQueue{cl::sycl::intel::fpga_emulator_selector{}};
...
```
