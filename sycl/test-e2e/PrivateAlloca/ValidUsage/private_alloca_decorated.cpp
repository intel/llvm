// RUN: %{build} -o %t.out
// RUN: echo 1  | %{run} %t.out
// RUN: echo 10 | %{run} %t.out
// RUN: echo 20 | %{run} %t.out
// RUN: echo 30 | %{run} %t.out

// Simple test filling a SYCL private alloca and copying it back to an output
// accessor using a decorated multi_ptr.

#include "Inputs/private_alloca_test.hpp"

constexpr sycl::specialization_id<int> size(10);

int main() {
  std::size_t n = 0;
  std::cin >> n;
  test<float, size, sycl::access::decorated::yes>(n);
  test<float, size, sycl::access::decorated::yes, alignof(float) * 2>(n);
}
