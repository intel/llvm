// RUN: %clang_cc1 %s -fsycl-is-device -fdeclare-spirv-builtins -fsyntax-only -emit-llvm -o - -O0 | FileCheck %s
//
// Check that SPIR-V builtins are declared with SYCL address spaces rather
// than OpenCL address spaces when using them with SYCL. OpenCL address spaces
// are mangled with the CL prefix and SYCL address spaces are mangled with the
// SY prefix.
//
// The opencl_global, opencl_local, and opencl_private attributes get turned
// into sycl_global, sycl_local and sycl_private address spaces by clang.

#include "Inputs/sycl.hpp"

// CHECK: __spirv_ocl_modf{{.*}}SYglobal
void modf_global(float a) {
  __attribute__((opencl_global)) float *ptr = nullptr;
  sycl::kernel_single_task<class fake_kernel>([=]() { __spirv_ocl_modf(a, ptr); });
}

// CHECK: __spirv_ocl_modf{{.*}}SYlocal
void modf_local(float a) {
  __attribute__((opencl_local)) float *ptr = nullptr;
  sycl::kernel_single_task<class fake_kernel>([=]() { __spirv_ocl_modf(a, ptr); });
}

// CHECK: __spirv_ocl_modf{{.*}}SYprivate
void modf_private(float a) {
  __attribute__((opencl_private)) float *ptr = nullptr;
  sycl::kernel_single_task<class fake_kernel>([=]() { __spirv_ocl_modf(a, ptr); });
}
