// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env UR_LAYER_ASAN_OPTIONS="detect_kernel_arguments:1" %{run} not %t 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  auto Data = sycl::malloc_device<int>(1, Q);
  ++Data;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { *Data = 0; });
  });
  Q.wait();
  // CHECK: ERROR: DeviceSanitizer: invalid-argument
  // CHECK: The {{[0-9]+}}th argument {{.*}} is located outside of its region

  return 0;
}
