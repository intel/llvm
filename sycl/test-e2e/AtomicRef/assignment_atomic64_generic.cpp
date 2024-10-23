// REQUIRES: aspect-atomic64
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// https://github.com/intel/llvm/issues/15791
// UNSUPPORTED: hip_amd

#include "assignment.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  device dev = q.get_device();

  const bool DoublesSupported = dev.has(sycl::aspect::fp64);

  constexpr int N = 32;
  if (DoublesSupported)
    assignment_generic_test<double>(q, N);

  // Include long tests if they are 64 bits wide
  if constexpr (sizeof(long) == 8) {
    assignment_generic_test<long>(q, N);
    assignment_generic_test<unsigned long>(q, N);
  }

  // Include long long tests if they are 64 bits wide
  if constexpr (sizeof(long long) == 8) {
    assignment_generic_test<long long>(q, N);
    assignment_generic_test<unsigned long long>(q, N);
  }

  // Include pointer tests if they are 64 bits wide
  if constexpr (sizeof(char *) == 8) {
    assignment_generic_test<char *>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
