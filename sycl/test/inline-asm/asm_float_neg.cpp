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

using dataType = cl::sycl::cl_float;

template <typename T = dataType>
struct KernelFunctor : WithInputBuffers<T, 1>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input) : WithInputBuffers<T, 1>(input), WithOutputBuffer<T>(input.size()) {}

  void operator()(cl::sycl::handler &cgh) {
    auto A = this->getInputBuffer().template get_access<cl::sycl::access::mode::read>(cgh);
    auto B = this->getOutputBuffer().template get_access<cl::sycl::access::mode::write>(cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        // clang-format off
        cl::sycl::range<1>{this->getOutputBufferSize()},
    [=](cl::sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
    // clang-format on
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
          asm("mov (M1, 8) %0(0, 0)<1> (-)%1(0, 0)<1;1,0>"
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
