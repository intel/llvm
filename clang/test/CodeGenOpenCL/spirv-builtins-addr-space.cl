// RUN: %clang_cc1 %s -x cl -fdeclare-spirv-builtins -fsyntax-only -emit-llvm -o - -O0 | FileCheck %s
//
// Check that SPIR-V builtins are declared with OpenCL address spaces rather
// than SYCL address spaces when using them with OpenCL. OpenCL address spaces
// are mangled with the CL prefix and SYCL address spaces are mangled with the
// SY prefix.

// CHECK:     __spirv_ocl_modffPU8CLglobal
void modf_global(float a, global float *ptr) { __spirv_ocl_modf(a, ptr); }

// CHECK:     __spirv_ocl_modffPU7CLlocal
void modf_local(float a, local float *ptr) { __spirv_ocl_modf(a, ptr); }

// CHECK:     __spirv_ocl_modffPU9CLprivate
void modf_private(float a) {
  float *ptr;
  __spirv_ocl_modf(a, ptr);
}
