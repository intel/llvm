// TODO fix windows failures
// UNSUPPORTED: windows && (level_zero || opencl)
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// tests sycl floating point math functions for sycl::vec and sycl::marray float
// and double cases.

#include "math_test_marray_vec_common.hpp"

int main() {
  queue deviceQueue;
  math_tests_4<float4>(deviceQueue);
  math_tests_4<marray<float, 4>>(deviceQueue);

  math_tests_3<float3>(deviceQueue);
  math_tests_3<marray<float, 3>>(deviceQueue);

  if (deviceQueue.get_device().has(sycl::aspect::fp64)) {
    math_tests_4<double4>(deviceQueue);
    math_tests_4<marray<double, 4>>(deviceQueue);

    math_tests_3<double3>(deviceQueue);
    math_tests_3<marray<double, 3>>(deviceQueue);
  }

  std::cout << "Pass" << std::endl;
  return 0;
}
