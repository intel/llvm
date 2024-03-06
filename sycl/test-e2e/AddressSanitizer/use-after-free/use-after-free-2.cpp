// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:5 UR_LOG_SANITIZER=level:info %{run} not %t &> %t.txt ; FileCheck --input-file %t.txt %s
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  auto *array =
      sycl::malloc_device<char>(1024, Q); // allocated size: 1280 <= 5120
  sycl::free(array, Q);

  // quarantine test
  auto *temp =
      sycl::malloc_device<char>(1024, Q); // allocated size: 1280*2 <= 5120
  sycl::free(temp, Q);
  temp = sycl::malloc_device<char>(1024, Q); // allocated size: 1280*3 <= 5120
  sycl::free(temp, Q);
  temp = sycl::malloc_device<char>(1024, Q); // allocated size: 1280*4 <= 5120
  sycl::free(temp, Q);
  // CHECK-NOT: <SANITIZER>[INFO]: Quarantine Free

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(1024, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK: ERROR: DeviceSanitizer: use-after-free on address [[ADDR:0x.*]]
  // CHECK: READ of size 1 at kernel <{{.*MyKernel}}>
  // CHECK:   #0 {{.*}} {{.*use-after-free-2.cpp}}:25
  // CHECK: [[ADDR]] is located inside of Device USM region [{{0x.*}}, {{0x.*}})
  // CHECK: allocated here:
  // CHECK:   {{#[0-9]+}} {{0x.*}} in main {{.*use-after-free-2.cpp}}:9
  // CHECK: released here:
  // CHECK:   {{#[0-9]+}} {{0x.*}} in main {{.*use-after-free-2.cpp}}:10

  return 0;
}
