// RUN: %{build} -w -o %t.out
// RUN: echo 10 20 30 | %{run} %t.out
// UNSUPPORTED: cuda || hip

// Chain of private_alloca test to check runtime support for compilation when
// the default size is to be used.

#include "Inputs/private_alloca_test.hpp"

constexpr sycl::specialization_id<std::size_t> size(10);
constexpr sycl::specialization_id<int> isize(10);
constexpr sycl::specialization_id<int16_t> ssize(100);

int main() {
  test<float, size, sycl::access::decorated::yes>();
  test<int16_t, isize, sycl::access::decorated::legacy>();
  test<int, ssize, sycl::access::decorated::no>();
}
