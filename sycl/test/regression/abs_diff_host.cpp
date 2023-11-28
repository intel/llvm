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
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  using RT = T;
#else
  using RT = std::make_unsigned_t<T>;
#endif

  constexpr T MaxVal = std::numeric_limits<T>::max();
  constexpr T MinVal = std::numeric_limits<T>::min();
  constexpr RT RMaxVal = MaxVal;

  // Sanity checks to make sure simple distances work.
  assert(sycl::abs_diff(MaxVal, T(10)) == RT(MaxVal - 10));
  assert(sycl::abs_diff(T(10), MaxVal) == RT(MaxVal - 10));
  assert(sycl::abs_diff(MinVal, T(-10)) == RT(MinVal - 10));
  assert(sycl::abs_diff(T(-10), MinVal) == RT(MinVal - 10));

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // Avoid unrepresentable negation.
  constexpr RT RAbsMinVal = RMaxVal - (MaxVal + MinVal);
  // Checks potential overflows. This is UB in SYCL 2020 Rev. 8.
  assert(sycl::abs_diff(MaxVal, T(-10)) == (RMaxVal + 10));
  assert(sycl::abs_diff(T(-10), MaxVal) == (RMaxVal + 10));
  assert(sycl::abs_diff(MinVal, T(10)) == (RAbsMinVal + 10));
  assert(sycl::abs_diff(T(10), MinVal) == (RAbsMinVal + 10));
  assert(sycl::abs_diff(MaxVal, MinVal) == (RMaxVal + RAbsMinVal));
  assert(sycl::abs_diff(MinVal, MaxVal) == (RMaxVal + RAbsMinVal));
#endif
}

int main() {
  check<char>();
  check<short>();
  check<int>();
  check<long>();
  check<long long>();
  return 0;
}
