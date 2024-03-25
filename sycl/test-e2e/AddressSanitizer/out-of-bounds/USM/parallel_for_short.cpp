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

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 123456789;
#if defined(MALLOC_HOST)
  auto *array = sycl::malloc_host<short>(N, Q);
#elif defined(MALLOC_SHARED)
  auto *array = sycl::malloc_shared<short>(N, Q);
#elif defined(MALLOC_DEVICE)
  auto *array = sycl::malloc_device<short>(N, Q);
#elif defined(MALLOC_SYSTEM)
  auto *array = new short[N];
#else
#error "Must provide malloc type to run the test"
#endif

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernelR_4>(
        sycl::nd_range<1>(N + 1, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK-DEVICE: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK-HOST:   ERROR: DeviceSanitizer: out-of-bounds-access on Host USM
  // CHECK-SHARED: ERROR: DeviceSanitizer: out-of-bounds-access on Shared USM
  // CHECK: {{READ of size 2 at kernel <.*MyKernelR_4> LID\(0, 0, 0\) GID\(123456789, 0, 0\)}}
  // CHECK: {{  #0 .* .*parallel_for_short.cpp:}}[[@LINE-7]]

  return 0;
}
