// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wimplicit-float-size-conversion -verify=size-only,always-size %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wimplicit-float-conversion -verify=always-size,precision-only %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wimplicit-float-conversion -Wno-implicit-float-size-conversion -verify=prefer-precision %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wno-implicit-float-conversion -verify %s

// This test checks that floating point conversion warnings are emitted correctly when used in conjunction.

#include "sycl.hpp"
class kernelA;

using namespace sycl;

int main() {
  queue q;
  // expected-no-diagnostics
  // always-size-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernelA, (lambda}}
  q.submit([&](handler &h) {
    h.single_task<class kernelA>([=]() {
      float s = 1.0; // always-size-warning {{implicit conversion between floating point types of different sizes}}
      // prefer-precision-warning@+2 {{implicit conversion loses floating-point precision: 'double' to 'float'}}
      // precision-only-warning@+1 {{implicit conversion loses floating-point precision: 'double' to 'float'}}
      float d = 2.1; // size-only-warning {{implicit conversion between floating point types of different sizes}}
    });
  });
  return 0;
}
