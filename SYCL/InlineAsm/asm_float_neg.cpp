// UNSUPPORTED: cuda || hip_nvidia
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "include/asmhelper.h"
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using dataType = sycl::cl_float;

template <typename T = dataType>
struct KernelFunctor : WithInputBuffers<T, 1>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input)
      : WithInputBuffers<T, 1>(input), WithOutputBuffer<T>(input.size()) {}

  void operator()(sycl::handler &cgh) {
    auto A =
        this->getInputBuffer().template get_access<sycl::access::mode::read>(
            cgh);
    auto B =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[intel::reqd_sub_group_size(16)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm("mov (M1, 16) %0(0, 0)<1> (-)%1(0, 0)<1;1,0>"
              : "=rw"(B[wiID])
              : "rw"(A[wiID]));
#else
          B[wiID] = -A[wiID];
#endif
        });
  }

  size_t problem_size = 0;
};

int main() {
  std::vector<dataType> input(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++)
    input[i] = 1.0 / i;

  KernelFunctor<> f(input);
  if (!launchInlineASMTest(f))
    return 0;

  auto &R = f.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; ++i) {
    if (R[i] != -input[i]) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << R[i] << " != " << -input[i] << "\n";
      return 1;
    }
  }

  return 0;
}
