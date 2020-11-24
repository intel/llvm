# DPC++ end-to-end tests for features under development

We require to have in-tree LIT tests independent from HW (e.g. GPU,
FPGA, etc) and external software (e.g OpenCL, Level Zero, CUDA runtimes, etc).

However, it might be helpful to have E2E tests stored in-tree for features
under active development if atomic change is required for the test and product.
This directory can contain such tests temporarily.

It is developer responsibility to move the tests from this directory to
[DPC++ E2E test suite](https://github.com/intel/llvm-test-suite/tree/intel/SYCL)
or [KhronosGroup/SYCL-CTS](https://github.com/KhronosGroup/SYCL-CTS) once the
feature is stabilized.
