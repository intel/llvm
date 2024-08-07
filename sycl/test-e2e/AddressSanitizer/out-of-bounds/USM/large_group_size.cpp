// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck --check-prefixes CHECK %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 12345678;
  auto *array = sycl::malloc_device<char>(N, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernelR_4>(
        sycl::nd_range<1>(N + 1, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK-DEVICE: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
  // CHECK: {{READ of size 1 at kernel <.*MyKernelR_4> LID\(0, 0, 0\) GID\(12345678, 0, 0\)}}
  // CHECK: {{  #0 .* .*large_group_size.cpp:}}[[@LINE-5]]

  sycl::free(array, Q);
  return 0;
}
