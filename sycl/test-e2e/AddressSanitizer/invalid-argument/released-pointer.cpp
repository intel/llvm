// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS="quarantine_size_mb:1;detect_kernel_arguments:1" %{run} not %t 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  auto Data = sycl::malloc_device<int>(1, Q);
  sycl::free(Data, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { *Data = 0; });
  });
  Q.wait();
  // CHECK: ERROR: DeviceSanitizer: invalid-argument
  // CHECK: The {{[0-9]+}}th argument {{.*}} is a released USM pointer
  // CHECK: {{.*}} is located inside of Device USM region
  // CHECK: allocated here:
  // CHECK: freed here:

  return 0;
}
