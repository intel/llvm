// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s
// This test checks if __builtin_assume_aligned does not throw an error when
// called from within device code.

#include "sycl.hpp"

using namespace sycl;
queue q;

int main() {
  int *Ptr[2];
  // expected-no-diagnostics
  q.submit([&](handler &h) {
    h.single_task<class kernelA>([=]() {
    int *APtr = (int *)__builtin_assume_aligned(Ptr, 32);
    *APtr = 42;
    });
  });
  return 0;
}

