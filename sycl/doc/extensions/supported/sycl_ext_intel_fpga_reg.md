# FPGA reg

Intel FPGA extension `fpga_reg()` is implemented in header file
`#include <CL/sycl/intel/fpga_extensions.hpp>`.

fpga_reg is used to help compiler infer that at least one register is on the corresponding data path.

## Implementation

The implementation is a wrapper class to map fpga_reg function call to a Clang built-in
\_\_builtin_intel_fpga_reg() only when parsing for SYCL device code.
```c++
#if __has_builtin(__builtin_intel_fpga_reg)
  return __builtin_intel_fpga_reg(t);
#else
  return t;
#endif

```


## Usage

```c++
#include <CL/sycl/intel/fpga_extensions.hpp>
...
// force at least one register on data path
int a = cl::sycl::intel::fpga_reg(a[k]) + b[k];

...
```

## Feature Test Macro

This extension provides a feature-test macro as described in the core SYCL
specification section 6.3.3 "Feature test macros". Therefore, an implementation
supporting this extension must predefine the macro `SYCL_EXT_INTEL_FPGA_REG`
to one of the values defined in the table below. Applications can test for the
existence of this macro to determine if the implementation supports this
feature, or applications can test the macro’s value to determine which of the
extension’s APIs the implementation supports.

|Value |Description|
|:---- |:---------:|
|1     |Initial extension version. Base features are supported.|
