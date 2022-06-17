// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -verify %s

#include "sycl.hpp"
class kernelA;

using namespace cl::sycl;

int main() {
  queue q;
  float *dst;
  q.submit([&](handler &h) {
    h.single_task<class kernelA>([=]() {
      dst[0] = 1.1; // expected-warning {{double precision arithmetic used in device code. reduced FP64 performance expected on GPU's.}}
      dst[1] = 0.0; // expected-warning {{double precision arithmetic used in device code. reduced FP64 performance expected on GPU's.}}
    });
  });

  return 0;
}
