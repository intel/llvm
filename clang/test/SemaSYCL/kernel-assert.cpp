// REQUIRES: system-windows
// RUN: %clang_cc1 -fsycl-is-device -triple nvptx64-nvidia-cuda -internal-isystem %S/Inputs/ -fsyntax-only -verify %s

// expected-no-diagnostics

#include <assert.h>

template <class T, class F>
__attribute__((sycl_kernel)) void kf(const F &) {
  volatile int a = 32;
  assert(a == 42);
}

int main() {
  kf<class X>([] {});
}
