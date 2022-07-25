// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs \
// RUN:   -sycl-std=2020 -verify -fsyntax-only %s

// This test checks whether we diagnose cases of unmarked, undefined
// __device__ functions called on device from either kernels or sycl
// device functions. This is needed because libdevice functions are
// declared but not defined in `__clang_cuda_libdevice_declares.h`.

#include "sycl.hpp"

__attribute__((device)) void undefined();

void fn(){ return undefined(); };

sycl::queue deviceQueue;

int main() {

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class CallToUndefinedFnTester>([]() {
      undefined();
      fn();
    });
  });
}
// expected-no-diagnostics
