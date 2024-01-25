// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Simple test for for size member and construction of an empty marray. It also
// tests that non-zero-sized marrays can be constructed with an empty marray as
// an argument.

#include <sycl/sycl.hpp>

int main() {
  sycl::marray<int, 0> EmptyMarray;
  static_assert(EmptyMarray.size() == 0);
  sycl::marray<int, 1> Marray1{EmptyMarray, 1};
  sycl::marray<int, 1> Marray2{1, EmptyMarray};
  return 0;
}
