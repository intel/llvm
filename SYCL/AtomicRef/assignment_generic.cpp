// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// CUDA backend has had no support for the generic address space yet
// XFAIL: cuda || hip

#include "assignment.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  constexpr int N = 32;
  assignment_generic_test<int>(q, N);
  assignment_generic_test<unsigned int>(q, N);
  assignment_generic_test<float>(q, N);

  // Include long tests if they are 32 bits wide
  if constexpr (sizeof(long) == 4) {
    assignment_generic_test<long>(q, N);
    assignment_generic_test<unsigned long>(q, N);
  }

  // Include pointer tests if they are 32 bits wide
  if constexpr (sizeof(char *) == 4) {
    assignment_generic_test<char *>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
