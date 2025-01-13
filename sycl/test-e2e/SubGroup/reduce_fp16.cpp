// REQUIRES: aspect-fp16
// REQUIRES: gpu

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
