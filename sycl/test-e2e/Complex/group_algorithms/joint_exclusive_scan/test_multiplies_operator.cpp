// REQUIRES: cuda

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include "joint_exclusive_scan.hpp"

int main() {
  bool result = true;

  /// Testing scalar with multiplies operator
  result &= test_scalar_joint_exclusive_scan<double, 4, sycl::multiplies<>>();
  result &= test_scalar_joint_exclusive_scan<float, 4, sycl::multiplies<>>();
  result &=
      test_scalar_joint_exclusive_scan<sycl::half, 4, sycl::multiplies<>>();

  /// Testing marray with multiplies operator
  result &= test_marray_joint_exclusive_scan<double, 4, sycl::multiplies<>>();
  result &= test_marray_joint_exclusive_scan<float, 4, sycl::multiplies<>>();
  result &=
      test_marray_joint_exclusive_scan<sycl::half, 4, sycl::multiplies<>>();

  return !result;
}
