// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out


#include "include/asmhelper.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#ifdef __SYCL_DEVICE_ONLY__
          #define CONSTANT __attribute__((opencl_constant))
#else
          #define CONSTANT
#endif

using dataType = cl::sycl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithOutputBuffer<T> {
  KernelFunctor(size_t problem_size) : WithOutputBuffer<T>(problem_size) {}

  void operator()(cl::sycl::handler &cgh) {
    cl::sycl::stream out(1024, 256, cgh);
    auto C = this->getOutputBuffer().template get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()}, [=](cl::sycl::id<1> wiID) [[cl::intel_reqd_sub_group_size(8)]] {
          int rand = 0;
           volatile int output = 0;
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
          asm volatile("cmp.eq (M1_NM, 1) P1 %1(0,0)<0;1,0> 0x0:d\n"
                       "(P1) sel (M1_NM, 1) %0(0,0)<1> 0x7:b 0x8:b"
                       : "=rw"(output) : "rw"(rand));
          
#else
          output = 7;
#endif
          C[wiID] = output;
        });
  }
};

int main() {
  KernelFunctor<> f(DEFAULT_PROBLEM_SIZE);
  if (!launchInlineASMTest(f))
    return 0;

  if (verify_all_the_same(f.getOutputBufferData(), 7))
    return 0;

  return 1;
}
