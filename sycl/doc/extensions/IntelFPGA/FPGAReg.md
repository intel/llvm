# FPGA reg

Intel FPGA extension `fpga_reg()` is implemented in header file
`#include <sycl/ext/intel/fpga_extensions.hpp>`.

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
#include <sycl/ext/intel/fpga_extensions.hpp>
...
// force at least one register on data path
int a = cl::sycl::intel::fpga_reg(a[k]) + b[k];

...
```
