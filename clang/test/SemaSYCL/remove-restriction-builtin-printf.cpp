// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s
// This test checks if __builtin_printf does not throw an error when
// called from within device code.

#include "sycl.hpp"

using namespace sycl;
queue q;

int main() {
  // expected-no-diagnostics
  q.submit([&](handler &h) {
    h.single_task<class kernelA>([=]() {
        __builtin_printf("hello, %d\n", 23);
    });
  });
  return 0;
}

