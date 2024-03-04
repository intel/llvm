// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:1 %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<char>(1024, Q);
  sycl::free(array, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(1024, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK-DEVICE: ERROR: DeviceSanitizer: use-after-free on address [[ADDR:0x.*]]
  // CHECK: READ of size 1 at kernel <{{.*MyKernel}}>
  // CHECK:   #0 {{.*}} {{.*use-after-free-1.cpp:}}[[@LINE-5]]
  // CHECK: [[ADDR]] is located inside of Device USM region [{{0x.*}}, {{0x.*}})
  // CHECK: allocated here:
  // CHECK:   {{#[0-9]+}} {{0x.*}} in main {{.*use-after-free-1.cpp:}}[[@LINE-14]]
  // CHECK: released here:
  // CHECK:   {{#[0-9]+}} {{0x.*}} in main {{.*use-after-free-1.cpp:}}[[@LINE-15]]

  return 0;
}
