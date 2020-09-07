// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// This test is disabled because there is a bug somewhere either in test or in
// the inline asm support
// FIXME: debug an issue and enable test back
// RUNx: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUNx: %t.out
// RUN: %clangxx -fsycl %s -o %t.ref.out
// RUN: %t.ref.out

#include "include/asmhelper.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using dataType = cl::sycl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithInputBuffers<T, 3>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input1, const std::vector<T> &input2, const std::vector<T> &input3) : WithInputBuffers<T, 3>(input1, input2, input3), WithOutputBuffer<T>(input1.size()) {}

  void operator()(cl::sycl::handler &cgh) {
    auto A = this->getInputBuffer(0).template get_access<cl::sycl::access::mode::read>(cgh);
    auto B = this->getInputBuffer(1).template get_access<cl::sycl::access::mode::read>(cgh);
    auto C = this->getInputBuffer(2).template get_access<cl::sycl::access::mode::read>(cgh);
    auto D = this->getOutputBuffer().template get_access<cl::sycl::access::mode::write>(cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()}, [=
    ](cl::sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
          asm("mad (M1, 8) %0(0, 0)<1> %3(0, 0)<1;1,0> %1(0, 0)<1;1,0> %2(0, 0)<1;1,0>"
              : "=rw"(D[wiID])
              : "rw"(A[wiID]), "rw"(B[wiID]), "rw"(C[wiID]));
#else
          D[wiID] = A[wiID] * B[wiID] + C[wiID];
#endif
        });
  }
};

int main() {
  std::vector<dataType> inputA(DEFAULT_PROBLEM_SIZE), inputB(DEFAULT_PROBLEM_SIZE), inputC(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    inputA[i] = i;
    inputB[i] = i;
    inputC[i] = DEFAULT_PROBLEM_SIZE - i * i;
  }

  KernelFunctor<> f(inputA, inputB, inputC);
  if (!launchInlineASMTest(f))
    return 0;

  auto &D = f.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; ++i) {
    if (D[i] != inputA[i] * inputB[i] + inputC[i]) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << D[i] << " != " << inputA[i] * inputB[i] + inputC[i] << "\n";
      return 1;
    }
  }
  return 0;
}
