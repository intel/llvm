// RUN: %{build} -w -o %t.out
// RUN: echo 1  | %{run} %t.out
// RUN: echo 10 | %{run} %t.out
// RUN: echo 20 | %{run} %t.out
// RUN: echo 30 | %{run} %t.out

// Simple test filling a private alloca and copying it back to an output
// accessor using a legacy multi_ptr.

#include "Inputs/private_alloca_test.hpp"

constexpr sycl::specialization_id<int16_t> size(10);

int main() {
  std::size_t n = 0;
  std::cin >> n;
  test<int, size, sycl::access::decorated::legacy>(n);
  test<int, size, sycl::access::decorated::legacy, alignof(int) * 4>(n);
}
