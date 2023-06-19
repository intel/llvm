// REQUIRES: aspect-fp64
// UNSUPPORTED: hip

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correct work of SPIR-V 1.3 reduce algorithm
// used with MUL operation.

#include "reduce.hpp"
#include <iostream>
int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check_mul<class MulDouble, double>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
