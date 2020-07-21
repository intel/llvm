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
        int const var_M = 714025;
        int const var_N = 2332;
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
        asm volatile("mul (M1, 8) %0(0, 0)<1> %1(0, 0)<1;1,0> %1(0, 0)<1;1,0>\n"
                     "add (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> %2(0, 0)<1;1,0>\n"
                     "mod (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> %3(0, 0)<0;1,0>\n"
                     "mul (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> 0x3E8:d\n"
                     "div (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> %4(0, 0)<0;1,0>\n"
                     "add (M1, 8) %0(0, 0)<1> %0(0, 0)<1;1,0> 0x1:d\n"
                      : "+rw"(C[wiID])
                      : "rw"(A[wiID]), "rw"(B[wiID]), "rw"(var_M), "rw"(var_N));
#else
        C[wiID] = A[wiID] * A[wiID];
        C[wiID] = C[wiID] + B[wiID];
        C[wiID] = C[wiID] % var_M;
        C[wiID] = C[wiID] * 1000;
        C[wiID] = C[wiID] / var_N;
        C[wiID] = C[wiID] + 1;
#endif
        });
  }
};

int main() {
  std::vector<dataType> inputA(DEFAULT_PROBLEM_SIZE), inputB(DEFAULT_PROBLEM_SIZE);
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    inputA[i] = i * 1111;
    inputB[i] = 5555555 * i;
  }

  KernelFunctor<> f(inputA, inputB);
  if (!launchInlineASMTest(f))
    return 0;

  auto &C = f.getOutputBufferData(); 
  
  int const var_M = 714025;
  int const var_N = 2332;
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    int intermediate_value = (inputA[i] * inputA[i] + inputB[i]) % var_M;
    intermediate_value = 1 + ( 1000 * intermediate_value) / var_N;
    if ( C[i] != intermediate_value) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << C[i] << " != " << intermediate_value << "\n";
      return 1;
    }
  }
  return 0;
}
