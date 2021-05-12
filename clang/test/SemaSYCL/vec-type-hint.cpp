// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2017 -fsyntax-only -DKERNEL= -verify=sycl-2017 %s
// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -fsyntax-only -DKERNEL= -verify=sycl-2020 %s
// RUN: %clang_cc1 -fsyntax-only -DKERNEL=kernel -verify=opencl -x cl %s

// opencl-no-diagnostics
// sycl-2017-no-diagnostics

// __attribute__((vec_type_hint)) is deprecated in OpenCL mode without
// replacement in SYCL 2020, but is allowed outside of OpenCL mode.
KERNEL __attribute__((vec_type_hint(int))) void foo() {} // sycl-2020-warning {{attribute 'vec_type_hint' is deprecated}}
