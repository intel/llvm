// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O1 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_HOST -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-HOST --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_SHARED -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-SHARED --input-file %t.txt %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

__attribute__((noinline)) void foo(int *array, size_t i) { array[i] = 1; }
// CHECK-DEVICE: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
// CHECK-HOST:   ERROR: DeviceSanitizer: out-of-bounds-access on Host USM
// CHECK-SHARED: ERROR: DeviceSanitizer: out-of-bounds-access on Shared USM
// CHECK: {{WRITE of size 4 at kernel <.*MyKernel> LID\(0, 0, 0\) GID\(123, 0, 0\)}}
// CHECK: {{  #0 foo\(int\*, unsigned long\) .*parallel_for_func.cpp:}}[[@LINE-5]]

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 123;
#if defined(MALLOC_HOST)
  auto *array = sycl::malloc_host<int>(N, Q);
#elif defined(MALLOC_SHARED)
  auto *array = sycl::malloc_shared<int>(N, Q);
#elif defined(MALLOC_DEVICE)
  auto *array = sycl::malloc_device<int>(N, Q);
#elif defined(MALLOC_SYSTEM)
  auto *array = new int[N];
#else
#error "Must provide malloc type to run the test"
#endif

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N + 1, 1),
        [=](sycl::nd_item<1> item) { foo(array, item.get_global_id(0)); });
  });
  Q.wait();

  return 0;
}
