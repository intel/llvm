// REQUIRES: linux
// RUN: %{build} %device_asan_flags -DMALLOC_DEVICE -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s
// RUN: %{build} %device_asan_flags -DMALLOC_DEVICE -O1 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s
// RUN: %{build} %device_asan_flags -DMALLOC_DEVICE -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK-DEVICE %s
// RUN: %{build} %device_asan_flags -DMALLOC_HOST -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK,CHECK-HOST %s
// RUN: %{build} %device_asan_flags -DMALLOC_SHARED -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-SHARED --input-file %t.txt %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 12;
#if defined(MALLOC_HOST)
  auto *array = sycl::malloc_host<char>(N, Q);
#elif defined(MALLOC_SHARED)
  auto *array = sycl::malloc_shared<char>(N, Q);
#else // defined(MALLOC_DEVICE)
  auto *array = sycl::malloc_device<char>(N, Q);
#endif

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(sycl::range<1>(N + 1),
                                   [=](sycl::id<1> i) { ++array[i]; });
  });
  Q.wait();
  // CHECK-DEVICE: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK-HOST:   ERROR: DeviceSanitizer: out-of-bounds-access on Host USM
  // CHECK-SHARED: ERROR: DeviceSanitizer: out-of-bounds-access on Shared USM
  // CHECK: READ of size 1 at kernel {{<.*MyKernel.*>}} LID({{.*}}, 0, 0) GID(12, 0, 0)
  // CHECK: {{  #0 .* .*parallel_no_local_size.cpp:}}[[@LINE-7]]

  sycl::free(array, Q);
  return 0;
}
