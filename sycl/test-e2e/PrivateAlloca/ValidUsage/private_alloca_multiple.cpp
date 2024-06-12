// RUN: %{build} -w -o %t.out
// RUN: echo 10 20 30 | %{run} %t.out

// Chain of private_alloca test to check runtime support for compilation when
// the default size is to be used.

#include "Inputs/private_alloca_test.hpp"

constexpr sycl::specialization_id<std::size_t> size(10);
constexpr sycl::specialization_id<int> isize(10);
constexpr sycl::specialization_id<int16_t> ssize(100);

int main() {
  constexpr std::size_t num_tests = 3;
  std::array<std::size_t, num_tests> ns;
  std::generate_n(ns.begin(), num_tests, []() {
    std::size_t i = 0;
    std::cin >> i;
    return i;
  });

  test<float, size, sycl::access::decorated::yes>(ns[0]);
  test<int16_t, isize, sycl::access::decorated::legacy>(ns[1]);
  test<int, ssize, sycl::access::decorated::no>(ns[2]);

  test<float, size, sycl::access::decorated::yes, alignof(float) * 2>(ns[0]);
  test<int16_t, isize, sycl::access::decorated::legacy, alignof(int16_t) * 2>(
      ns[1]);
  test<int, ssize, sycl::access::decorated::no, alignof(int) * 2>(ns[2]);
}
