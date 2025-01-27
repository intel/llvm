// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: %{run} %t 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 12;
  auto *array = sycl::malloc_device<char>(N, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernelR_4>(
        sycl::nd_range<1>(N, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  }).wait();
  // CHECK: ERROR: DeviceSanitizer: detected memory leaks of Device USM
  // CHECK: Direct leak of 12 byte(s) at {{0x.*}} allocated from:
  // CHECK: in main {{.*memory-leak.cpp:}}[[@LINE-9]]

  return 0;
}
