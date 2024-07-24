// REQUIRES: linux
// RUN: %{build} %device_asan_flags -DUNSAFE -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=redzone:64 %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -DSAFE -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LOG_SANITIZER=level:debug UR_LAYER_ASAN_OPTIONS=redzone:8 %{run} %t 2>&1 | FileCheck --check-prefixes CHECK-MIN %s
// RUN: env SYCL_PREFER_UR=1 UR_LOG_SANITIZER=level:debug UR_LAYER_ASAN_OPTIONS=max_redzone:4096 %{run} %t 2>&1 | FileCheck --check-prefixes CHECK-MAX %s

#include <sycl/usm.hpp>

int main() {
  sycl::queue q;
  constexpr std::size_t N = 8;
  auto *array = sycl::malloc_device<char>(N, q);

  q.submit([&](sycl::handler &h) {
#ifdef UNSAFE
     h.single_task<class Test>([=]() { ++array[N + 24]; });
#else
     h.single_task<class Test>([=]() { ++array[0]; });
#endif
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 1 at kernel <.*Test> LID\(0, 0, 0\) GID\(0, 0, 0\)}}
  // CHECK: {{  #0 .* .*config-red-zone-size.cpp:}}[[@LINE-7]]
  // CHECK-MIN: Trying to set redzone size to a value less than 16 is ignored
  // CHECK-MAX: Trying to set max redzone size to a value greater than 2048 is ignored

  sycl::free(array, q);
  return 0;
}
