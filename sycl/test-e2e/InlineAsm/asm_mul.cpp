// UNSUPPORTED: cuda, hip
// REQUIRES: gpu,linux,sg-16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "include/asmhelper.h"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <vector>

using dataType = sycl::opencl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithInputBuffers<T, 2>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input1, const std::vector<T> &input2)
      : WithInputBuffers<T, 2>(input1, input2), WithOutputBuffer<T>(
                                                    input1.size()) {}
  void operator()(sycl::handler &cgh) {
    auto A =
        this->getInputBuffer(0).template get_access<sycl::access::mode::read>(
            cgh);
    auto B =
        this->getInputBuffer(1).template get_access<sycl::access::mode::read>(
            cgh);
    auto C =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[sycl::reqd_sub_group_size(16)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm("mul (M1, 16) %0(0, 0)<1> %1(0, 0)<1;1,0> %2(0, 0)<1;1,0>"
              : "=rw"(C[wiID])
              : "rw"(A[wiID]), "rw"(B[wiID]));
#else
          C[wiID] = A[wiID] * B[wiID];
#endif
        });
  }
};

int main() {
  std::vector<dataType> inputA(DEFAULT_PROBLEM_SIZE),
      inputB(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    inputA[i] = i;
    inputB[i] = DEFAULT_PROBLEM_SIZE - i;
  }

  KernelFunctor<> f(inputA, inputB);
  if (!launchInlineASMTest(f, {16}))
    return 0;

  auto &C = f.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; ++i) {
    if (C[i] != inputA[i] * inputB[i]) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << C[i] << " != " << inputA[i] * inputB[i] << "\n";
      return 1;
    }
  }

  return 0;
}
