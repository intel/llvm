// REQUIRES: aspect-fp64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correct work of the sub-group algorithm reduce().

#include "reduce.hpp"

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<class KernelName_alTnImqzYasRyHjYg, double>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
