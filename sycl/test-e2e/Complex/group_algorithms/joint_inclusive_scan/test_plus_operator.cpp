// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include "joint_inclusive_scan.hpp"

int main() {
  bool result = true;

  /// Testing scalar with plus operator
  result &= test_scalar_joint_inclusive_scan<double, 4, sycl::plus<>>();
  result &= test_scalar_joint_inclusive_scan<float, 4, sycl::plus<>>();
  result &= test_scalar_joint_inclusive_scan<sycl::half, 4, sycl::plus<>>();

  /// Testing marray with plus operator
  result &= test_marray_joint_inclusive_scan<double, 4, sycl::plus<>>();
  result &= test_marray_joint_inclusive_scan<float, 4, sycl::plus<>>();
  result &= test_marray_joint_inclusive_scan<sycl::half, 4, sycl::plus<>>();

  return !result;
}
