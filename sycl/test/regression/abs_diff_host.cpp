// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes %s -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %t2.out %}

// Test checks that sycl::abs_diff correctly handles signed operations that
// might overflow.

#include <sycl/sycl.hpp>

#include <limits>
#include <type_traits>

template <typename T> void check() {
  constexpr T MaxVal = std::numeric_limits<T>::max();
  constexpr T MinVal = std::numeric_limits<T>::min();

  // Sanity checks to make sure simple distances work.
  assert(sycl::abs_diff(MaxVal, T(10)) == T(MaxVal - 10));
  assert(sycl::abs_diff(T(10), MaxVal) == T(MaxVal - 10));
  assert(sycl::abs_diff(MinVal, T(-10)) == T(MinVal - 10));
  assert(sycl::abs_diff(T(-10), MinVal) == T(MinVal - 10));
}

int main() {
  check<char>();
  check<short>();
  check<int>();
  check<long>();
  check<long long>();
  return 0;
}
