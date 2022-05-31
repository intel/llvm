// REQUIRES: system-windows
// RUN: %clang_cc1 -fsycl-is-device -triple nvptx64-nvidia-cuda -internal-isystem %S/Inputs/ -fsyntax-only -verify %s

// expected-no-diagnostics

#include <assert.h>

// Make sure that SYCL kernels can correctly call asserts on Windows. We do not
// expect to see any warnings or compilation errors. On Windows targeting NVPTX
// the implementation of assert is provided by libclc.
//
// Please note, that the execution of the assert is tested in llvm-test-suite.
template <class T, class F>
__attribute__((sycl_kernel)) void kf(const F &) {
  volatile int a = 32;
  assert(a == 42);
}

int main() {
  kf<class X>([] {});
}
