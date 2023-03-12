// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs \
// RUN:   -sycl-std=2020 -verify -fsyntax-only %s

// This test checks whether we diagnose cases of unmarked, undefined
// __attribute__((device)) (a.k.a __device__) functions called on
// device from either kernels or sycl device functions. This is
// needed because libdevice functions are declared but not defined
// in `__clang_cuda_libdevice_declares.h`.
// A check on __attribute__((sycl_device)) (a.k.a SYCL_EXTERNAL) has
// been introduced as well.

#include "sycl.hpp"
#include "../CodeGenCUDA/Inputs/cuda.h"

__device__ void cuda_dev_undefined_0();
__attribute__((device)) void cuda_dev_undefined_1();
void fn_0(){
  cuda_dev_undefined_0();
  cuda_dev_undefined_1();
}

__attribute__((sycl_device)) void sycl_ext_undefined();
void fn_1(){ sycl_ext_undefined(); }

sycl::queue deviceQueue;

int main() {

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class CallToUndefinedFnTester>([]() {
      cuda_dev_undefined_0();
      cuda_dev_undefined_1();
      sycl_ext_undefined();
      fn_0();
    });
  });
}
// expected-no-diagnostics
