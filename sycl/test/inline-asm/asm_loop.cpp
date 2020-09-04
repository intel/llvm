// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl %s -o %t.ref.out
// RUN: %GPU_RUN_PLACEHOLDER %t.ref.out

#include "include/asmhelper.h"
#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using DataType = cl::sycl::cl_int;

template <typename T = DataType>
struct KernelFunctor : WithInputBuffers<T, 2>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input1, const std::vector<T> &input2)
      : WithInputBuffers<T, 2>(input1, input2), WithOutputBuffer<T>(
                                                    input1.size()) {}

  void operator()(cl::sycl::handler &CGH) {
    auto A = this->getInputBuffer(0)
                 .template get_access<cl::sycl::access::mode::read>(CGH);
    auto B = this->getInputBuffer(1)
                 .template get_access<cl::sycl::access::mode::read>(CGH);
    auto C = this->getOutputBuffer()
                 .template get_access<cl::sycl::access::mode::write>(CGH);
    // clang-format off
    CGH.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()},
    [=](cl::sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
// clang-format on
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
          asm volatile(".decl P1 v_type=P num_elts=8\n"
                       ".decl P2 v_type=P num_elts=8\n"
                       ".decl temp v_type=G type=d num_elts=8 align=dword\n"
                       "mov (M1, 8) %0(0, 0)<1> 0x0:d\n"
                       "cmp.le (M1, 8) P1 %1(0,0)<1;1,0> 0x0:d\n"
                       "(P1) goto (M1, 8) label0\n"
                       "mov (M1, 8) temp(0,0)<1> 0x0:d\n"
                       "label1:\n"
                       "add (M1, 8) temp(0,0)<1> temp(0,0)<1;1,0> 0x1:w\n"
                       "add (M1, 8) %0(0,0)<1> %0(0,0)<1;1,0> %2(0,0)<1;1,0>\n"
                       "cmp.lt (M1, 8) P2 temp(0,0)<0;8,1> %1(0,0)<0;8,1>\n"
                       "(P2) goto (M1, 8) label1\n"
                       "label0:"
                       : "+rw"(C[wiID])
                       : "rw"(A[wiID]), "rw"(B[wiID]));
#else
          C[wiID] = 0;
          for (int i = 0; i < A[wiID]; ++i) {
            C[wiID] = C[wiID] + B[wiID];
          }
#endif
        });
  }
};

int main() {
  std::vector<DataType> InputA(DEFAULT_PROBLEM_SIZE),
      InputB(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    InputA[i] = i;
    InputB[i] = 2 * i;
  }

  KernelFunctor<> Functor(InputA, InputB);
  if (!launchInlineASMTest(Functor))
    return 0;

  auto &C = Functor.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    if (C[i] != InputA[i] * InputB[i]) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << C[i] << " != " << InputA[i] * InputB[i] << "\n";
      return 1;
    }
  }

  return 0;
}
