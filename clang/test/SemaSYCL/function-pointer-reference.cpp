// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -fsyntax-only -verify %s
//
// Test which checks that lvalue reference to function pointer does not
// crash

// expected-no-diagnostics
#include "Inputs/sycl.hpp"

int f() { return 0; }

void foo() {
  cl::sycl::kernel_single_task<class Kernel>([=]() {
    int (*p)() = f;
    int (&r)() = *p;
  });
}
