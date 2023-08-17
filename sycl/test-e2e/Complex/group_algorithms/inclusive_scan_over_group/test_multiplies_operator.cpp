// REQUIRES: cuda

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include "inclusive_scan_over_group.hpp"

int main() {
  bool result = true;

  /// Testing scalar with multiplies operator
  result &=
      test_scalar_inclusive_scan_over_group<double, 4, sycl::multiplies<>>();
  result &=
      test_scalar_inclusive_scan_over_group<float, 4, sycl::multiplies<>>();
  result &= test_scalar_inclusive_scan_over_group<sycl::half, 4,
                                                  sycl::multiplies<>>();

  /// Testing marray with multiplies operator
  result &=
      test_marray_inclusive_scan_over_group<double, 4, sycl::multiplies<>>();
  result &=
      test_marray_inclusive_scan_over_group<float, 4, sycl::multiplies<>>();
  result &= test_marray_inclusive_scan_over_group<sycl::half, 4,
                                                  sycl::multiplies<>>();

  return !result;
}
