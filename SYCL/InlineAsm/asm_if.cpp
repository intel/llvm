// UNSUPPORTED: cuda || hip_nvidia
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "include/asmhelper.h"
#include <sycl/sycl.hpp>

using DataType = cl::sycl::cl_int;

template <typename T = DataType> struct KernelFunctor : WithOutputBuffer<T> {
  KernelFunctor(size_t ProblemSize) : WithOutputBuffer<T>(ProblemSize) {}

  void operator()(cl::sycl::handler &CGH) {
    auto C = this->getOutputBuffer()
                 .template get_access<cl::sycl::access::mode::write>(CGH);
    bool switchField = false;
    CGH.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()}, [=
    ](cl::sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
          int Output = 0;
#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile("{\n"
                       ".decl P1 v_type=P num_elts=1\n"
                       "cmp.eq (M1_NM, 8) P1 %1(0,0)<0;1,0> 0x0:b\n"
                       "(P1) sel (M1_NM, 8) %0(0,0)<1> 0x7:d 0x8:d"
                       "}\n"
                       : "=rw"(Output)
                       : "rw"(switchField));

#else
          if (switchField == false)
            Output = 7;
          else
            Output = 8;
#endif
          C[wiID] = Output;
        });
  }
};

int main() {
  KernelFunctor<> Functor(DEFAULT_PROBLEM_SIZE);
  if (!launchInlineASMTest(Functor))
    return 0;

  if (verify_all_the_same(Functor.getOutputBufferData(), 7))
    return 0;

  return 1;
}
