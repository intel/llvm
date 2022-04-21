// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -fsyntax-only -Wno-sycl-2017-compat -verify %s
//
// This test checks if compiler reports compilation error on an attempt to pass
// a pointer to VLA as kernel argument

#include "Inputs/sycl.hpp"

void foo(unsigned X) {
  using VLATy = float(*)[X];
  VLATy PtrToVLA;
  cl::sycl::kernel_single_task<class Kernel>([=]() {
    // expected-error@+1 {{variable length arrays are not supported for the current target}}
    (void)PtrToVLA;
  });
}
