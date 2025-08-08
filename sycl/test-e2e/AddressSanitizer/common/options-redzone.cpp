// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -DUNSAFE -O0 -g -o %t1.out
// RUN: env UR_LAYER_ASAN_OPTIONS=redzone:4000 %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -DSAFE -O0 -g -o %t2.out

// clang-format off
// RUN: env UR_LOG_SANITIZER=level:debug UR_LAYER_ASAN_OPTIONS=redzone:8 %{run} %t2.out 2>&1 | FileCheck --check-prefixes CHECK-MIN %s
// clang-format on

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
  // CHECK: <SANITIZER>[WARNING]: Increasing the redzone size may cause excessive memory overhead
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 1 at kernel <.*Test> LID\(0, 0, 0\) GID\(0, 0, 0\)}}
  // CHECK: {{  #0 .* .*options-redzone.cpp:}}[[@LINE-8]]
  // CHECK-MIN: The valid range of "redzone" is [16, 18446744073709551615]. Setting to the minimum value 16.

  sycl::free(array, q);
  return 0;
}
