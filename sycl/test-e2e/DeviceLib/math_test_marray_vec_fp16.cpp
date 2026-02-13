// REQUIRES: aspect-fp16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// tests sycl floating point math functions for sycl::vec and sycl::marray fp16
// cases.

#include "math_test_marray_vec_common.hpp"

int main() {
  queue deviceQueue;

  math_tests_4<half4>(deviceQueue);
  math_tests_4<marray<half, 4>>(deviceQueue);
  math_tests_3<half3>(deviceQueue);
  math_tests_3<marray<half, 3>>(deviceQueue);

  std::cout << "Pass" << std::endl;
  return 0;
}
