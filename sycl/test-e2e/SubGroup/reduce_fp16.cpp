// REQUIRES: aspect-fp16
// REQUIRES: gpu

// XFAIL: windows && gpu-intel-gen12
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21533

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correct work of the sub-group algorithm reduce().

#include "reduce.hpp"

int main() {
  queue Queue;
  check<class KernelName_oMg, sycl::half>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
