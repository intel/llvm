// REQUIRES: linux
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O1 -g -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_DEVICE -O2 -g -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-DEVICE --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_HOST -O2 -g -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-HOST --input-file %t.txt %s
// RUN: %{build} %device_sanitizer_flags -DMALLOC_SHARED -O2 -g -o %t
// RUN: %{run} not %t &> %t.txt ; FileCheck --check-prefixes CHECK,CHECK-SHARED --input-file %t.txt %s
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 12345;
#if defined(MALLOC_HOST)
  auto *array = sycl::malloc_host<char>(N, Q);
#elif defined(MALLOC_SHARED)
  auto *array = sycl::malloc_shared<char>(N, Q);
#elif defined(MALLOC_DEVICE)
  auto *array = sycl::malloc_device<char>(N, Q);
#elif defined(MALLOC_SYSTEM)
  auto *array = new char[N];
#else
#error "Must provide malloc type to run the test"
#endif

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernelR_4>(
        sycl::nd_range<1>(N + 1, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK-DEVICE: ERROR: DeviceSanitizer: out-of-bounds-access on USM Device Memory
  // CHECK-HOST:   ERROR: DeviceSanitizer: out-of-bounds-access on USM Host Memory
  // CHECK-SHARED: ERROR: DeviceSanitizer: out-of-bounds-access on USM Shared Memory
  // CHECK: {{READ of size 1 at kernel <.*MyKernelR_4> LID\(0, 0, 0\) GID\(12345, 0, 0\)}}
  // CHECK: {{  #0 .* .*parallel_for_char.cpp:}}[[@LINE-8]]

  return 0;
}
