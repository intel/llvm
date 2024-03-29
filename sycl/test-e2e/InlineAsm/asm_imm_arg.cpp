// UNSUPPORTED: cuda, hip
// REQUIRES: gpu,linux,sg-16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "include/asmhelper.h"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <vector>

constexpr int CONST_ARGUMENT = 0xabc;
using dataType = sycl::opencl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithInputBuffers<T, 1>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input)
      : WithInputBuffers<T, 1>(input), WithOutputBuffer<T>(input.size()) {}

  void operator()(sycl::handler &cgh) {
    auto A =
        this->getInputBuffer(0).template get_access<sycl::access::mode::read>(
            cgh);
    auto B =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[sycl::reqd_sub_group_size(16)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm("add (M1, 16) %0(0, 0)<1> %1(0, 0)<1;1,0> %2"
              : "=rw"(B[wiID])
              : "rw"(A[wiID]), "i"(CONST_ARGUMENT));
#else
          B[wiID] = A[wiID] + CONST_ARGUMENT;
#endif
        });
  }
};

int main() {
  std::vector<dataType> input(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++)
    input[i] = i;

  KernelFunctor<> f(input);
  if (!launchInlineASMTest(f, {16}))
    return 0;

  auto &B = f.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; ++i) {
    if (B[i] != input[i] + CONST_ARGUMENT) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << B[i] << " != " << input[i] + CONST_ARGUMENT << "\n";
      return 1;
    }
  }
  return 0;
}
