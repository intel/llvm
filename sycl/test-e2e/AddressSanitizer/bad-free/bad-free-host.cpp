// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: %force_device_asan_rt %{run} not %t 2>&1 | FileCheck %s
#include <sycl/usm.hpp>

constexpr size_t N = 64;

int main() {
  sycl::queue Q;
  auto *data = new int[N];
  sycl::free(data, Q);
  return 0;
}
// CHECK: ERROR: DeviceSanitizer: bad-free on address [[ADDR:0x.*]]
// CHECK: [[ADDR]] may be allocated on Host Memory
