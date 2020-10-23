// This is a basic acceptance test for inline ASM feature. More tests can be
// found in https://github.com/intel/llvm-test-suite/tree/intel/SYCL/InlineAsm
// RUN: %clangxx -fsycl %s -o %t.out

#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>
#include <vector>

constexpr const size_t DEFAULT_PROBLEM_SIZE = 16;

using DataType = sycl::cl_int;

int main() {
  DataType DataA[DEFAULT_PROBLEM_SIZE], DataB[DEFAULT_PROBLEM_SIZE],
      DataC[DEFAULT_PROBLEM_SIZE];
  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    DataA[i] = i;
    DataB[i] = 2 * i;
  }

  // Create a simple asynchronous exception handler.
  auto AsyncHandler = [](sycl::exception_list ExceptionList) {
    for (auto &Exception : ExceptionList) {
      std::rethrow_exception(Exception);
    }
  };

  {
    sycl::buffer<DataType, 1> BufA(DataA, DEFAULT_PROBLEM_SIZE);
    sycl::buffer<DataType, 1> BufB(DataB, DEFAULT_PROBLEM_SIZE);
    sycl::buffer<DataType, 1> BufC(DataC, DEFAULT_PROBLEM_SIZE);

    sycl::queue deviceQueue(sycl::gpu_selector{}, AsyncHandler);

    deviceQueue.submit([&](sycl::handler &cgh) {
      auto A = BufA.get_access<sycl::access::mode::read>(cgh);
      auto B = BufB.get_access<sycl::access::mode::read>(cgh);
      auto C = BufC.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<class FillBuffer>(
          sycl::range<1>{DEFAULT_PROBLEM_SIZE}, [=
      ](sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
#if defined(__SYCL_DEVICE_ONLY__)
            asm volatile(
                ".decl P1 v_type=P num_elts=8\n"
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
    });
  }

  for (int i = 0; i < DEFAULT_PROBLEM_SIZE; i++) {
    if (DataC[i] != DataA[i] * DataB[i]) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << DataC[i] << " != " << DataA[i] * DataB[i] << "\n";
      return 1;
    }
  }

  return 0;
}
