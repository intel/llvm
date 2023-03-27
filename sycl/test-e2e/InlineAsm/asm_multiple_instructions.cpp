// UNSUPPORTED: cuda || hip
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl -DTO_PASS %s -o %t.out.pass
// RUN: %GPU_RUN_PLACEHOLDER %t.out.pass
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// The test is failing when writing directly to output buffer.
// If temporary variable is used (see TO_PASS mode) the test succeeded.
// XFAIL: gpu
#include "include/asmhelper.h"
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using dataType = sycl::opencl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithInputBuffers<T, 3>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input1, const std::vector<T> &input2,
                const std::vector<T> &input3)
      : WithInputBuffers<T, 3>(input1, input2, input3), WithOutputBuffer<T>(
                                                            input1.size()) {}

  void operator()(sycl::handler &cgh) {
    auto A = this->getInputBuffer(0)
                 .template get_access<sycl::access::mode::read_write>(cgh);
    auto B =
        this->getInputBuffer(1).template get_access<sycl::access::mode::read>(
            cgh);
    auto C =
        this->getInputBuffer(2).template get_access<sycl::access::mode::read>(
            cgh);
    auto D =
        this->getOutputBuffer().template get_access<sycl::access::mode::write>(
            cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        sycl::range<1>{this->getOutputBufferSize()},
        [=](sycl::id<1> wiID) [[intel::reqd_sub_group_size(16)]] {
#if defined(TO_PASS)
          // The code below passing verification
          volatile int output = -1;

#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile(
              "{\n"
              "add (M1, 16) %1(0, 0)<1> %1(0, 0)<1;1,0> %2(0, 0)<1;1,0>\n"
              "add (M1, 16) %1(0, 0)<1> %1(0, 0)<1;1,0> %3(0, 0)<1;1,0>\n"
              "mov (M1, 16) %0(0, 0)<1> %1(0, 0)<1;1,0>\n"
              "}\n"
              : "=rw"(output), "+rw"(A[wiID])
              : "rw"(B[wiID]), "rw"(C[wiID]));
#else
          A[wiID] += B[wiID];
          A[wiID] += C[wiID];
          output = A[wiID];
#endif
          D[wiID] = output;
#else
#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile(
              "{\n"
              "add (M1, 16) %1(0, 0)<1> %1(0, 0)<1;1,0> %2(0, 0)<1;1,0>\n"
              "add (M1, 16) %1(0, 0)<1> %1(0, 0)<1;1,0> %3(0, 0)<1;1,0>\n"
              "mov (M1, 16) %0(0, 0)<1> %1(0, 0)<1;1,0>\n"
              "}\n"
              : "=rw"(D[wiID]), "+rw"(A[wiID])
              : "rw"(B[wiID]), "rw"(C[wiID]));
#else
          A[wiID] += B[wiID];
          A[wiID] += C[wiID];
          D[wiID] = A[wiID];
#endif
#endif
        });
  }
};

int main() {
  std::vector<dataType> inputA(DEFAULT_PROBLEM_SIZE),
      inputB(DEFAULT_PROBLEM_SIZE), inputC(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    inputA[i] = inputB[i] = i;
    inputC[i] = DEFAULT_PROBLEM_SIZE - 2 * i; // A[i] + B[i] + C[i] = LIST_SIZE
  }

  KernelFunctor<> f(inputA, inputB, inputC);
  if (!launchInlineASMTest(f))
    return 0;

  if (verify_all_the_same(f.getOutputBufferData(),
                          (dataType)DEFAULT_PROBLEM_SIZE))
    return 0;

  return 1;
}
