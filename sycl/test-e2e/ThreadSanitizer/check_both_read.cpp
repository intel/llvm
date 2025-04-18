// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_tsan_flags -O0 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
#include "sycl/detail/core.hpp"
#include "sycl/usm.hpp"

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<char>(1, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class MyKernelR_4>(
         sycl::nd_range<1>(32, 8),
         [=](sycl::nd_item<1> item) { auto value = array[0]; });
   }).wait();
  // CHECK-NOT: WARNING: DeviceSanitizer: data race

  sycl::free(array, Q);
  return 0;
}
