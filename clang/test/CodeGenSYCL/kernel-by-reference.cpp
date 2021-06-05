// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -verify -fsyntax-only -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -verify -fsyntax-only -sycl-std=2020 -DSYCL2020 %s
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -verify -fsyntax-only -Wno-sycl-strict -DNODIAG %s
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -verify -fsyntax-only -sycl-std=2020 -Wno-sycl-strict -DNODIAG %s

#include "sycl.hpp"

using namespace cl::sycl;

int simple_add(int i) {
  return i + 1;
}

// ensure both compile.
int main() {
  queue q;
#if defined(SYCL2020)
  // expected-warning@#KernelSingleTask2017 {{passing kernel functions by value is deprecated in SYCL 2020}}
  // expected-note@+3 {{in instantiation of function template specialization}}
#endif
  q.submit([&](handler &h) {
    h.single_task_2017<class sycl2017>([]() { simple_add(10); });
  });

#if defined(SYCL2017)
  // expected-warning@#KernelSingleTask {{passing of kernel functions by reference is a SYCL 2020 extension}}
  // expected-note@+3 {{in instantiation of function template specialization}}
#endif
  q.submit([&](handler &h) {
    h.single_task<class sycl2020>([]() { simple_add(11); });
  });

  return 0;
}
#if defined(NODIAG)
// expected-no-diagnostics
#endif
