// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wimplicit-float-size-conversion -verify -DCHECK_SIZE %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wimplicit-float-conversion -verify -DCHECK_PRECISION %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wno-implicit-float-conversion -DCHECK_NO_WARNING -verify %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -Wimplicit-float-conversion -Wno-implicit-float-size-conversion -verify -DCHECK_PRECISION_WITHOUT_SIZE %s

#include "sycl.hpp"
class kernelA;

using namespace cl::sycl;

int main() {
  queue q;
#if defined(CHECK_SIZE)
  q.submit([&](handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernelA, (lambda}}
    h.single_task<class kernelA>([=]() {
      float s = 1.0; // expected-warning {{implicit conversion between floating point types of different sizes}}
      float d = 2.1; // expected-warning {{implicit conversion between floating point types of different sizes}}
    });
  });

#elif defined(CHECK_PRECISION)
  q.submit([&](handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernelA, (lambda}}
    h.single_task<class kernelA>([=]() {
      float s = 1.0; // expected-warning {{implicit conversion between floating point types of different sizes}}
      float d = 2.1; // expected-warning {{implicit conversion loses floating-point precision: 'double' to 'float'}}
    });
  });

#elif defined(CHECK_NO_WARNING)
  // expected-no-diagnostics
  q.submit([&](handler &h) {
    h.single_task<class kernelA>([=]() {
      float s = 1.0;
      float d = 2.1;
    });
  });

#elif defined(CHECK_PRECISION_WITHOUT_SIZE)
  q.submit([&](handler &h) {
    h.single_task<class kernelA>([=]() {
      float s = 1.0;
      float d = 2.1; // expected-warning {{implicit conversion loses floating-point precision: 'double' to 'float'}}
    });
  });

#endif
  return 0;
}
