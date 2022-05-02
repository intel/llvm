# FPGA selector

Intel FPGA users can use header file: `#include<sycl/ext/intel/fpga_device_selector.hpp>` to simplify their code
when they want to specify FPGA hardware device or FPGA emulation device.

## Implementation

Current implementation is based on platform name. This is useful in the most common case when user have
one FPGA board installed in their system (one device per platform). 

## Usage: select FPGA hardware device
```c++
#include <sycl/ext/intel/fpga_device_selector.hpp>
...
// force FPGA hardware device
cl::sycl::queue deviceQueue{cl::sycl::ext::intel::fpga_selector{}};
...
```

## Usage: select FPGA emulator device
```c++
#include <sycl/ext/intel/fpga_device_selector.hpp>
...
// force FPGA emulation device
cl::sycl::queue deviceQueue{cl::sycl::ext::intel::fpga_emulator_selector{}};
...
```

## Feature Test Macro

This extension provides a feature-test macro as described in the core SYCL
specification section 6.3.3 "Feature test macros". Therefore, an implementation
supporting this extension must predefine the macro `SYCL_EXT_INTEL_FPGA_DEVICE_SELECTOR`
to one of the values defined in the table below. Applications can test for the
existence of this macro to determine if the implementation supports this
feature, or applications can test the macro’s value to determine which of the
extension’s APIs the implementation supports.

|Value |Description|
|:---- |:---------:|
|1     |Initial extension version. Base features are supported.|
