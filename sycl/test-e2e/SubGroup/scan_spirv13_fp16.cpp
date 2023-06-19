// REQUIRES: aspect-fp16
// REQUIRES: gpu
// UNSUPPORTED: hip

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correct work of SPIR-V 1.3 exclusive_scan() and
// inclusive_scan() algoriths used with the MUL operation.

#include "scan.hpp"
#include <iostream>

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check_mul<class MulHalf, sycl::half>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
