// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_ENABLE_LAYERS=UR_LAYER_ASAN ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_HOST -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_ENABLE_LAYERS=UR_LAYER_ASAN ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-HOST --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_SHARED -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_ENABLE_LAYERS=UR_LAYER_ASAN ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-SHARED --input-file %t.txt %s
#include <sycl/sycl.hpp>

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
// CHECK: ERROR: DeviceSanitizer: bad-free on [[ADDR:0x.*]]
// CHECK: {{#[0-9]+}} {{0x.*}} in main {{.*bad-free-2.cpp}}:[[@LINE-4]]
// CHECK-HOST:   [[ADDR]] is located inside of Host USM region {{\[0x.*, 0x.*\)}}
// CHECK-SHARED: [[ADDR]] is located inside of Shared USM region {{\[0x.*, 0x.*\)}}
// CHECK-DEVICE: [[ADDR]] is located inside of Device USM region {{\[0x.*, 0x.*\)}}
// CHECK-HOST:   {{#[0-9]+}} {{0x.*}} in main {{.*bad-free-2.cpp}}:[[@LINE-15]]
// CHECK-SHARED: {{#[0-9]+}} {{0x.*}} in main {{.*bad-free-2.cpp}}:[[@LINE-14]]
// CHECK-DEVICE: {{#[0-9]+}} {{0x.*}} in main {{.*bad-free-2.cpp}}:[[@LINE-13]]
