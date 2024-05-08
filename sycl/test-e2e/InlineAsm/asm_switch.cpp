// UNSUPPORTED: cuda || hip
// REQUIRES: gpu,linux,sg-16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "include/asmhelper.h"
#include <sycl/detail/core.hpp>

using DataType = sycl::opencl::cl_int;

template <typename T = DataType> struct KernelFunctor : WithOutputBuffer<T> {
  KernelFunctor(size_t ProblemSize) : WithOutputBuffer<T>(ProblemSize) {}

  void operator()(sycl::handler &CGH) {
    auto C =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            CGH);
    int switchField = 2;
    CGH.parallel_for<KernelFunctor<T>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[sycl::reqd_sub_group_size(16)]] {
          int Output = 0;
#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile("{\n"
                       ".decl P1 v_type=P num_elts=1\n"
                       ".decl P2 v_type=P num_elts=1\n"
                       ".decl P3 v_type=P num_elts=1\n"
                       "cmp.ne (M1_NM, 16) P1 %1(0,0)<0;1,0> 0x0:d\n"
                       "(P1) goto (M1, 1) label0%=\n"
                       "mov (M1, 8) %0(0,0)<1> 0x9:d\n"
                       "(P1) goto (M1, 1) label0%=\n"
                       "label0%=:\n"
                       "cmp.ne (M1_NM, 16) P2 %1(0,0)<0;1,0> 0x1:d\n"
                       "(P2) goto (M1, 1) label1%=\n"
                       "mov (M1, 8) %0(0,0)<1> 0x8:d\n"
                       "label1%=:\n"
                       "cmp.ne (M1_NM, 16) P3 %1(0,0)<0;1,0> 0x2:d\n"
                       "(P3) goto (M1, 1) label2%=\n"
                       "mov (M1, 16) %0(0,0)<1> 0x7:d\n"
                       "label2%=:"
                       "}\n"
                       : "=rw"(Output)
                       : "rw"(switchField));

#else
          switch (switchField) {
          case 0:
            Output = 9;
            break;
          case 1:
            Output = 8;
            break;
          case 2:
            Output = 7;
            break;
          }
#endif
          C[wiID] = Output;
        });
  }
};

int main() {
  KernelFunctor<> Functor(DEFAULT_PROBLEM_SIZE);
  if (!launchInlineASMTest(Functor, {16}))
    return 0;

  if (verify_all_the_same(Functor.getOutputBufferData(), 7))
    return 0;

  return 1;
}
