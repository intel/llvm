// UNSUPPORTED: cuda || hip_nvidia
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "include/asmhelper.h"
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using dataType = sycl::opencl::cl_int;

template <typename T = dataType> struct KernelFunctor : WithOutputBuffer<T> {
  KernelFunctor(size_t problem_size) : WithOutputBuffer<T>(problem_size) {}

  void operator()(sycl::handler &cgh) {
    auto C =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            cgh);
    cgh.parallel_for<KernelFunctor<T>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[intel::reqd_sub_group_size(16)]] {
          C[wiID] = 43;
#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile("");
#endif
        });
  }
};

int main() {
  KernelFunctor<> f(DEFAULT_PROBLEM_SIZE);
  if (!launchInlineASMTest(f))
    return 0;

  if (verify_all_the_same(f.getOutputBufferData(), 43))
    return 0;

  return 1;
}
