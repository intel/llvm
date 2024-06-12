// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: %force_device_asan_rt %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s
// RUN: %{build} %device_asan_flags -DMALLOC_HOST -O0 -g -o %t
// RUN: %force_device_asan_rt %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK-HOST %s
// RUN: %{build} %device_asan_flags -DMALLOC_SHARED -O0 -g -o %t
// RUN: %force_device_asan_rt %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK-SHARED %s
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

  sycl::free(data - 1, Q);
  return 0;
}
// CHECK: ERROR: DeviceSanitizer: bad-free on address [[ADDR:0x.*]]
// CHECK-HOST:   [[ADDR]] is located inside of Host USM region {{\[0x.*, 0x.*\)}}
// CHECK-SHARED: [[ADDR]] is located inside of Shared USM region {{\[0x.*, 0x.*\)}}
// CHECK-DEVICE: [[ADDR]] is located inside of Device USM region {{\[0x.*, 0x.*\)}}
