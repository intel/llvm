// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "assignment.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  constexpr int N = 32;
  assignment_test<int>(q, N);
  assignment_test<unsigned int>(q, N);
  assignment_test<float>(q, N);

  // Include long tests if they are 32 bits wide
  if constexpr (sizeof(long) == 4) {
    assignment_test<long>(q, N);
    assignment_test<unsigned long>(q, N);
  }

  // Include pointer tests if they are 32 bits wide
  if constexpr (sizeof(char *) == 4) {
    assignment_test<char *>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
