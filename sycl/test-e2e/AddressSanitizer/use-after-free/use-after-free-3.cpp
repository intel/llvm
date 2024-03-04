// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu UR_ENABLE_LAYERS=UR_LAYER_ASAN UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:5 UR_LOG_SANITIZER=level:debug %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --input-file %t.txt %s
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  auto *array =
      sycl::malloc_device<char>(1024, Q); // allocated size: 1280 <= 5120
  sycl::free(array, Q);
  // CHECK: <SANITIZER>[DEBUG]: ==== urUSMFree: [[ADDR1:0x.*]]

  auto *temp =
      sycl::malloc_device<char>(1024, Q); // allocated size: 1280*2 <= 5120
  sycl::free(temp, Q);
  // CHECK: <SANITIZER>[DEBUG]: ==== urUSMFree: [[ADDR2:0x.*]]

  temp = sycl::malloc_device<char>(1024, Q); // allocated size: 1280*3 <= 5120
  sycl::free(temp, Q);
  // CHECK: <SANITIZER>[DEBUG]: ==== urUSMFree: [[ADDR3:0x.*]]

  temp = sycl::malloc_device<char>(1024, Q); // allocated size: 1280*4 <= 5120
  sycl::free(temp, Q);
  // CHECK: <SANITIZER>[DEBUG]: ==== urUSMFree: [[ADDR4:0x.*]]

  temp = sycl::malloc_device<char>(1024, Q); // allocated size: 1280*5 > 5120
  sycl::free(temp, Q);
  // CHECK: <SANITIZER>[DEBUG]: ==== urUSMFree: [[ADDR5:0x.*]]
  // CHECK: <SANITIZER>[INFO]: Quarantine Free: [[ADDR1]]

  return 0;
}
