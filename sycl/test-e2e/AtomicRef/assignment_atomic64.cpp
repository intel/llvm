// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "assignment.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  device dev = q.get_device();

  if (!dev.has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  const bool DoublesSupported = dev.has(sycl::aspect::fp64);

  constexpr int N = 32;
  if (DoublesSupported)
    assignment_test<double>(q, N);

  // Include long tests if they are 64 bits wide
  if constexpr (sizeof(long) == 8) {
    assignment_test<long>(q, N);
    assignment_test<unsigned long>(q, N);
  }

  // Include long long tests if they are 64 bits wide
  if constexpr (sizeof(long long) == 8) {
    assignment_test<long long>(q, N);
    assignment_test<unsigned long long>(q, N);
  }

  // Include pointer tests if they are 64 bits wide
  if constexpr (sizeof(char *) == 8) {
    assignment_test<char *>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
