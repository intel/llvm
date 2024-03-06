// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_ENABLE_LAYERS=UR_LAYER_ASAN UR_LAYER_ASAN_OPTIONS=quarantine_size_mb:5 UR_LOG_SANITIZER=level:info %{run} not %t &> %t.txt ; FileCheck --input-file %t.txt %s
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  auto *array =
      sycl::malloc_device<char>(1024, Q); // allocated size: 1280 <= 5120
  // CHECK: AllocBegin=[[ADDR1:0x[0-9a-f]+]]
  sycl::free(array, Q);

  auto *temp =
      sycl::malloc_device<char>(1024, Q); // allocated size: 1280*2 <= 5120
  // CHECK: AllocBegin=[[ADDR2:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  temp = sycl::malloc_device<char>(1024, Q); // allocated size: 1280*3 <= 5120
  // CHECK: AllocBegin=[[ADDR3:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  temp = sycl::malloc_device<char>(1024, Q); // allocated size: 1280*4 <= 5120
  // CHECK: AllocBegin=[[ADDR4:0x[0-9a-f]+]]
  sycl::free(temp, Q);

  temp = sycl::malloc_device<char>(1024, Q); // allocated size: 1280*5 > 5120
  // CHECK: AllocBegin=[[ADDR5:0x[0-9a-f]+]]
  sycl::free(temp, Q);
  // CHECK: Quarantine Free: [[ADDR1]]

  return 0;
}
