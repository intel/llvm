// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_tsan_flags -O0 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_tsan_flags -O2 -g -o %t2.out
// RUN: %{run} %t2.out 2>&1 | FileCheck %s
#include "sycl/detail/core.hpp"
#include "sycl/usm.hpp"

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 32;
  auto *array = sycl::malloc_device<char>(N, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class MyKernelR_4>(
         sycl::nd_range<1>(N, 8),
         [=](sycl::nd_item<1> item) { array[item.get_group_linear_id()]++; });
   }).wait();
  // CHECK-NOT: WARNING: DeviceSanitizer: data race

  sycl::free(array, Q);
  return 0;
}
