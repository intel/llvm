// REQUIRES: aspect-fp64

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correct work of SPIR-V 1.3 reduce algorithm
// used with MUL operation.

#include "reduce.hpp"
#include <iostream>
int main() {
  queue Queue;
  check_mul<class MulDouble, double>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
