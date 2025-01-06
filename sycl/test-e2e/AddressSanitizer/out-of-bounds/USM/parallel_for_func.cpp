// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -DMALLOC_DEVICE -O0 -g -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s
// RUN: %{build} %device_asan_flags -DMALLOC_DEVICE -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s
// RUN: %{build} %device_asan_flags -DMALLOC_DEVICE -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s
// RUN: %{build} %device_asan_flags -DMALLOC_HOST -O2 -g -o %t4.out
// RUN: %{run} not %t4.out 2>&1 | FileCheck --check-prefixes CHECK,CHECK-HOST %s
// RUN: %{build} %device_asan_flags -DMALLOC_SHARED -O2 -g -o %t5.out
// RUN: %{run} not %t5.out &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-SHARED --input-file %t.txt %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

__attribute__((noinline)) void foo(int *array, size_t i) { array[i] = 1; }
// CHECK-DEVICE: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
// CHECK-HOST:   ERROR: DeviceSanitizer: out-of-bounds-access on Host USM
// CHECK-SHARED: ERROR: DeviceSanitizer: out-of-bounds-access on Shared USM
// CHECK: {{WRITE of size 4 at kernel <.*MyKernel> LID\(0, 0, 0\) GID\(12, 0, 0\)}}
// CHECK: {{  #0 foo\(int\*, unsigned long\) .*parallel_for_func.cpp:}}[[@LINE-5]]

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 12;
#if defined(MALLOC_HOST)
  auto *array = sycl::malloc_host<int>(N, Q);
#elif defined(MALLOC_SHARED)
  auto *array = sycl::malloc_shared<int>(N, Q);
#else // defined(MALLOC_DEVICE)
  auto *array = sycl::malloc_device<int>(N, Q);
#endif

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N + 1, 1),
        [=](sycl::nd_item<1> item) { foo(array, item.get_global_id(0)); });
  });
  Q.wait();

  sycl::free(array, Q);
  return 0;
}
