// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -verify %s

// This test warns users when an ODR-use of a constexpr variable causes the kernel lambda to capture it as a
// kernel argument

#include "sycl.hpp"

using namespace cl::sycl;

queue q;

class LambdaKernel;

int main() {

  constexpr unsigned OdrUsedVar = 10;

  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<LambdaKernel>(
        [=]() {
          // constexpr 'OdrUsedVar' is odr-used here.
          const unsigned *ptr = &OdrUsedVar; // expected-warning {{captured constexpr 'OdrUsedVar' will be a kernel argument in device code}}
        });
  });
  return 0;
}
