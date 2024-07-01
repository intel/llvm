// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -Xarch_device -fsanitize-recover=address -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} %t 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 1024;
  auto *array = sycl::malloc_device<int>(N, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Kernel>(
         sycl::nd_range<1>(N + 20, 1),
         [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
   }).wait();
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK: ====ERROR: DeviceSanitizer
  // CHECK-NOT: ====ERROR: DeviceSanitizer

  return 0;
}
