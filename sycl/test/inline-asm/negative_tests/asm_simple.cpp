// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: not %t.out 2>&1 | FileCheck %s

// CHECK: syntax error

#include "../include/asmhelper.h"
#include <CL/sycl.hpp>

using dataType = cl::sycl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithOutputBuffer<T> {
  KernelFunctor(size_t problem_size) : WithOutputBuffer<T>(problem_size) {}

  void operator()(cl::sycl::handler &cgh) {
    cgh.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()}, [=](cl::sycl::id<1> wiID) [[cl::intel_reqd_sub_group_size(8)]] {          
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
          asm volatile(".decl tmp1 v_type=G type=d num_elts=16 align=GRF\n"
                       ".decl tmp2 v_type=G type=d num_elts=16 align=GRF\n"
                       "@@\n");
#endif
        });
  }
};

int main() {
  KernelFunctor<> f(DEFAULT_PROBLEM_SIZE);
  launchInlineASMTest(f);
  return 0;
}
