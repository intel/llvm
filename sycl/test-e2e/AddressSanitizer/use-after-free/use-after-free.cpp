// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:1 %{run} not %t 2>&1 | FileCheck %s
#include <sycl/usm.hpp>

constexpr size_t N = 1024;

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<char>(N, Q);
  sycl::free(array, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK: ERROR: DeviceSanitizer: use-after-free on address [[ADDR:0x.*]]
  // CHECK: READ of size 1 at kernel <{{.*MyKernel}}>
  // CHECK:   #0 {{.*}} {{.*use-after-free.cpp:}}[[@LINE-5]]
  // CHECK: [[ADDR]] is located inside of Device USM region [{{0x.*}}, {{0x.*}})
  // CHECK: allocated here:
  // CHECK: released here:

  return 0;
}
