// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -DKERNEL= %s
// RUN: %clang_cc1 -fsyntax-only -DKERNEL=kernel -verify=opencl -x cl %s

// opencl-no-diagnostics

// __attribute__((vec_type_hint)) is deprecated without replacement in SYCL 2020 mode, but
// is allowed in OpenCL mode.
KERNEL __attribute__((vec_type_hint(int))) void foo() {} // sycl-2020-warning {{attribute 'vec_type_hint' is deprecated; attribute ignored}}
