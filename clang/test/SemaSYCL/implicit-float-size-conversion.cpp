// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wimplicit-float-size-conversion -verify %s

#include "sycl.hpp"
class kernelA;

using namespace cl::sycl;

int main() {
  queue q;
  q.submit([&](handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernelA, (lambda}}
    h.single_task<class kernelA>([=]() {
      float s = 1.0; // expected-warning {{implicit conversion between floating point types of different sizes}}
      float d = 2.1; // no size warning.
    });
  });

  return 0;
}