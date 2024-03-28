// UNSUPPORTED: true
// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_ENABLE_LAYERS=UR_LAYER_ASAN %{run-unfiltered-devices} not %t 2>&1 | FileCheck --check-prefixes CHECK %s
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q1{sycl::gpu_selector_v}, Q2{sycl::cpu_selector_v};
  auto *data = sycl::malloc_device<char>(64, Q1);
  sycl::free(data, Q2);
  return 0;
}
// CHECK: ERROR: DeviceSanitizer: bad-context on address [[ADDR:0x.*]]
// CHECK: {{#[0-9]+}} {{0x.*}} in main {{.*mismatched-queue-1.cpp}}:[[@LINE-4]]
// CHECK: [[ADDR]] is located inside of Device USM region {{\[0x.*, 0x.*\)}}
// CHECK:   {{#[0-9]+}} {{0x.*}} in main {{.*mismatched-queue-1.cpp}}:[[@LINE-7]]
