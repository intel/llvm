// REQUIRES: usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test ensures the result computed by exclusive_scan_over_group
// for the first work item when given a short or char argument with
// the maximum or minimum operator is computed correctly.
#include "../helpers.hpp"
#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;
queue q;
int cur_test = 0;
int n_fail = 0;

template <typename T, typename OpT> void test() {
  auto op = OpT();
  auto init = sycl::known_identity_v<decltype(op), T>;
  auto *p = malloc_shared<T>(1, q);
  *p = 0;
  T ref;
  emu::exclusive_scan(p, p + 1, &ref, init, op);
  range r(1);
  q.parallel_for(nd_range(r, r), [=](nd_item<1> it) {
     auto g = it.get_group();
     *p = exclusive_scan_over_group(g, *p, op);
   }).wait();

  if (*p != ref) {
    std::cout << "test " << cur_test << " fail\n";
    std::cout << "got:      " << int(*p) << "\n";
    std::cout << "expected: " << int(ref) << "\n\n";
    ++n_fail;
  }
  ++cur_test;
  free(p, q);
}

int main() {
  test<char, sycl::maximum<char>>();
  test<signed char, sycl::maximum<signed char>>();
  test<unsigned char, sycl::maximum<unsigned char>>();
  test<char, sycl::maximum<void>>();
  test<signed char, sycl::maximum<void>>();
  test<unsigned char, sycl::maximum<void>>();
  test<short, sycl::maximum<short>>();
  test<unsigned short, sycl::maximum<unsigned short>>();
  test<short, sycl::maximum<void>>();
  test<unsigned short, sycl::maximum<void>>();

  test<char, sycl::minimum<char>>();
  test<signed char, sycl::minimum<signed char>>();
  test<unsigned char, sycl::minimum<unsigned char>>();
  test<char, sycl::minimum<void>>();
  test<signed char, sycl::minimum<void>>();
  test<unsigned char, sycl::minimum<void>>();
  test<short, sycl::minimum<short>>();
  test<unsigned short, sycl::minimum<unsigned short>>();
  test<short, sycl::minimum<void>>();
  test<unsigned short, sycl::minimum<void>>();
  return n_fail != 0;
}
