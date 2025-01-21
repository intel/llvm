// REQUIRES: aspect-fp64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correct work of the sub-group algorithm reduce().

#include "reduce.hpp"

int main() {
  queue Queue;
  check<class KernelName_alTnImqzYasRyHjYg, double>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
