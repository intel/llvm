// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include "reduce_over_group.hpp"

int main() {
  bool result = true;

  /// Testing scalar with plus operator
  result &= test_scalar_reduce_over_group<double, 4, sycl::plus<>>();
  result &= test_scalar_reduce_over_group<float, 4, sycl::plus<>>();
  result &= test_scalar_reduce_over_group<sycl::half, 4, sycl::plus<>>();

  /// Testing marray with plus operator
  result &= test_marray_reduce_over_group<double, 4, sycl::plus<>>();
  result &= test_marray_reduce_over_group<float, 4, sycl::plus<>>();
  result &= test_marray_reduce_over_group<sycl::half, 4, sycl::plus<>>();

  return !result;
}
