// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -aux-triple x86_64-unknown-linux-gnu -Wno-sycl-2017-compat -verify -fsyntax-only  %s

#include "sycl.hpp"

sycl::queue deviceQueue;

int main(int argc, char **argv) {
  //_mm_prefetch is an x86-64 specific builtin where the second integer parameter is required to be a constant
  //between 0 and 7.
  _mm_prefetch("test", 4); // no error thrown, since this is a valid invocation

  _mm_prefetch("test", 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<AName, (lambda}}
    h.single_task<class AName>([]() {
      _mm_prefetch("test", 4); // expected-error {{builtin is not supported on this target}}
      _mm_prefetch("test", 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}} expected-error {{builtin is not supported on this target}}
    });
  });

  return 0;
}
