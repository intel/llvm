// REQUIRES: linux, (gpu && level_zero), cpu
// RUN: %{build} %device_asan_flags -DMALLOC_DEVICE -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS="detect_kernel_arguments:1" %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue gpu_queue(sycl::gpu_selector_v);
  sycl::queue cpu_queue(sycl::cpu_selector_v);

  auto data = sycl::malloc_device<int>(1, cpu_queue);

  gpu_queue.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { *data = 0; });
  });
  gpu_queue.wait();
  // CHECK: DeviceSanitizer: invalid-argument on kernel
  // CHECK: The {{[0-9]+}}th argument {{.*}} is allocated in other context
  // CHECK: {{.*}} is located inside of Device USM region

  sycl::free(data, cpu_queue);
  return 0;
}
