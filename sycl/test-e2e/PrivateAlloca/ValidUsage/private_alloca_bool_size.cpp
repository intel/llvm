// RUN: %{build} -w -o %t.out
// RUN: echo 1  | %{run} %t.out

// Test checking size of 'bool' type. This is not expected to be ever used, but,
// as 'bool' is an integral type, it is a possible scenario.

#include "Inputs/private_alloca_test.hpp"

constexpr sycl::specialization_id<bool> size(true);

int main() {
  std::size_t n = 0;
  std::cin >> n;
  test<int, size, sycl::access::decorated::legacy>(n);
  test<int, size, sycl::access::decorated::legacy, alignof(int) * 2>(n);
}
