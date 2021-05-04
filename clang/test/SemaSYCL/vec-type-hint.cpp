// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -DKERNEL= -verify=sycl %s
// RUN: %clang_cc1 -fsyntax-only -DKERNEL=kernel -verify=opencl -x cl %s

// opencl-no-diagnostics

// __attribute__((vec_type_hint)) is deprecated in OpenCL mode without
// replacement, but is allowed outside of OpenCL mode.
KERNEL __attribute__((vec_type_hint(int))) void foo() {} // sycl-warning {{attribute 'vec_type_hint' is deprecated}}
