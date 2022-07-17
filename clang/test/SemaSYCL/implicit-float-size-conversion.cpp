// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wimplicit-float-size-conversion -verify %s

#include "sycl.hpp"
class kernelA;

using namespace cl::sycl;

int main() {
  queue q;
  q.submit([&](handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernelA, (lambda}}
    h.single_task<class kernelA>([=]() {
      float ff = 2.1;
      double d = 2.0f;
    });
  });

  return 0;
}
