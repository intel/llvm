// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

// Test checks that sycl::abs_diff correctly handles signed operations that
// might overflow.

#include <sycl/sycl.hpp>

#include <limits>
#include <type_traits>

template <typename T> void check() {
  using UT = std::make_unsigned_t<T>;
  constexpr T MaxVal = std::numeric_limits<T>::max();
  constexpr T MinVal = std::numeric_limits<T>::min();
  constexpr UT UMaxVal = MaxVal;
  // Avoid unrepresentable negation.
  constexpr UT UAbsMinVal = UMaxVal - (MaxVal + MinVal);

  // Sanity checks to make sure simple distances work.
  assert(sycl::abs_diff(MaxVal, T(10)) == UT(MaxVal - 10));
  assert(sycl::abs_diff(T(10), MaxVal) == UT(MaxVal - 10));
  assert(sycl::abs_diff(MinVal, T(-10)) == UT(MinVal - 10));
  assert(sycl::abs_diff(T(-10), MinVal) == UT(MinVal - 10));

  // Checks potential overflows.
  assert(sycl::abs_diff(MaxVal, T(-10)) == (UMaxVal + 10));
  assert(sycl::abs_diff(T(-10), MaxVal) == (UMaxVal + 10));
  assert(sycl::abs_diff(MinVal, T(10)) == (UAbsMinVal + 10));
  assert(sycl::abs_diff(T(10), MinVal) == (UAbsMinVal + 10));
  assert(sycl::abs_diff(MaxVal, MinVal) == (UMaxVal + UAbsMinVal));
  assert(sycl::abs_diff(MinVal, MaxVal) == (UMaxVal + UAbsMinVal));
}

int main() {
  check<char>();
  check<short>();
  check<int>();
  check<long>();
  check<long long>();
  return 0;
}
