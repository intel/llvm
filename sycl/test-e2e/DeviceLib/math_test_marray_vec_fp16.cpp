// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// tests sycl floating point math functions for sycl::vec and sycl::marray fp16
// cases.

#include "math_test_marray_vec_common.hpp"

int main() {
  queue deviceQueue;

  if (!deviceQueue.get_device().has(sycl::aspect::fp16)) {
    std::cout << "skipping fp16 tests: requires fp16 device aspect."
              << std::endl;
    return 0;
  }
  math_tests_4<half4>(deviceQueue);
  math_tests_4<marray<half, 4>>(deviceQueue);
  math_tests_3<half3>(deviceQueue);
  math_tests_3<marray<half, 3>>(deviceQueue);

  std::cout << "Pass" << std::endl;
  return 0;
}
