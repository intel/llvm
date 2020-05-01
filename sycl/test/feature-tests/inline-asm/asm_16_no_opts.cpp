// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out
// RUN: %clangxx -fsycl %s -o %t.ref.out
// RUN: %t.ref.out

#include "include/asmhelper.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using dataType = cl::sycl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithOutputBuffer<T> {
  KernelFunctor(size_t problem_size) : WithOutputBuffer<T>(problem_size) {}

  void operator()(cl::sycl::handler &cgh) {
    auto C = this->getOutputBuffer().template get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()}, [=](cl::sycl::id<1> wiID) [[cl::intel_reqd_sub_group_size(16)]] {
          for (int i = 0; i < 10; ++i) {
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
            asm("fence_sw");
            C[wiID] += i;

#else
            C[wiID] += i;
#endif
          }
        });
  }
};

int main() {
  KernelFunctor<> f(DEFAULT_PROBLEM_SIZE);
  if (!launchInlineASMTest(f))
    return 0;

  if (verify_all_the_same(f.getOutputBufferData(), 45))
    return 0;

  return 1;
}
