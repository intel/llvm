// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out
// RUN: %clangxx -fsycl %s -o %t.ref.out
// RUN: %t.ref.out

#include "include/asmhelper.h"
#include <CL/sycl.hpp>
#include <cmath>

using dataType = cl::sycl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithInputBuffers<T, 2>, WithOutputBuffer<T> {
  KernelFunctor(const std::vector<T> &input1, const std::vector<T> &input2) : WithInputBuffers<T, 2>(input1, input2), WithOutputBuffer<T>(input1.size()) {}

  void operator()(cl::sycl::handler &cgh) {
    auto A = this->getInputBuffer(0).template get_access<cl::sycl::access::mode::read>(cgh);
    auto B = this->getInputBuffer(1).template get_access<cl::sycl::access::mode::read>(cgh);
    auto C = this->getOutputBuffer().template get_access<cl::sycl::access::mode::write>(cgh);

    cgh.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()}, [=](cl::sycl::id<1> wiID) [[cl::intel_reqd_sub_group_size(8)]] {
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
        asm volatile("add (M1, 8) %0(0, 0)<1> %1(0, 0)<1;1,0> %2(0, 0)<1;1,0>\n"
                     "mul (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> %1(0, 0)<1;1,0>\n"
                     "add (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> (-)%2(0, 0)<1;1,0>\n"
                     "mul (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> 0x4:d\n"
                     "add (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> 0x64:d\n"
                     "mul (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> %0(0, 0)<1;1,0>\n"
                      : "+rw"(C[wiID])
                      : "rw"(A[wiID]), "rw"(B[wiID]));
#else
        C[wiID] = A[wiID] + B[wiID];
        C[wiID] = C[wiID] * A[wiID];
        C[wiID] = C[wiID] - B[wiID];
        C[wiID] = C[wiID] * 4;
        C[wiID] = C[wiID] + 100;
        C[wiID] = C[wiID] * C[wiID];
#endif
        });
  }
};

int main() {
  std::vector<dataType> inputA(DEFAULT_PROBLEM_SIZE), inputB(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    inputA[i] = i;
    inputB[i] = 2 * i;
  }

  KernelFunctor<> f(inputA, inputB);
  if (!launchInlineASMTest(f))
    return 0;

  auto &C = f.getOutputBufferData();
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    int intermediate_value = std::pow(((((inputA[i] + inputB[i]) * inputA[i]) - inputB[i]) * 4) + 100, 2);
    if ( C[i] != intermediate_value) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << C[i] << " != " << intermediate_value << "\n";
      return 1;
    }
  }

  return 0;
}
