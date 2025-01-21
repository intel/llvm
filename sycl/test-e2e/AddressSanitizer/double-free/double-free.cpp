// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O0 -g -o %t1.out
// RUN: %force_device_asan_rt UR_LAYER_ASAN_OPTIONS="quarantine_size_mb:1;detect_kernel_arguments:0" %{run} not %t1.out 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s
// RUN: %{build} %device_asan_flags -DMALLOC_HOST -O0 -g -o %t2.out
// RUN: %force_device_asan_rt UR_LAYER_ASAN_OPTIONS="quarantine_size_mb:1;detect_kernel_arguments:0" %{run} not %t2.out 2>&1 | FileCheck --check-prefixes CHECK,CHECK-HOST %s
// RUN: %{build} %device_asan_flags -DMALLOC_SHARED -O0 -g -o %t3.out
// RUN: %force_device_asan_rt UR_LAYER_ASAN_OPTIONS="quarantine_size_mb:1;detect_kernel_arguments:0" %{run} not %t3.out 2>&1 | FileCheck --check-prefixes CHECK,CHECK-SHARED %s
#include <sycl/usm.hpp>

constexpr size_t N = 64;

int main() {
  sycl::queue Q;

#if defined(MALLOC_HOST)
  auto *data = sycl::malloc_host<int>(N, Q);
#elif defined(MALLOC_SHARED)
  auto *data = sycl::malloc_shared<int>(N, Q);
#else
  auto *data = sycl::malloc_device<int>(N, Q);
#endif

  sycl::free(data, Q);
  sycl::free(data, Q);

  return 0;
}
// CHECK: ERROR: DeviceSanitizer: double-free on address [[ADDR:0x.*]]
// CHECK-HOST:   [[ADDR]] is located inside of Host USM region {{\[0x.*, 0x.*\)}}
// CHECK-SHARED: [[ADDR]] is located inside of Shared USM region {{\[0x.*, 0x.*\)}}
// CHECK-DEVICE: [[ADDR]] is located inside of Device USM region {{\[0x.*, 0x.*\)}}
// CHECK: freed here
// CHECH: in main {{.*double-free.cpp:}}[@LINE-33]
// CHECK: previously allocated here
// CHECK-HOST: in main {{.*double-free.cpp:}}[[@LINE-19]]
// CHECK-SHARED: in main {{.*double-free.cpp:}}[[@LINE-18]]
// CHECK-DEVICE: in main {{.*double-free.cpp:}}[[@LINE-17]]
