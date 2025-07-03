// REQUIRES: linux, cpu || (gpu && level_zero)
// ALLOW_RETRIES: 10
// RUN: %{build} %device_tsan_flags -O2 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
#include "sycl/detail/core.hpp"
#include "sycl/usm.hpp"
#include "sycl/vector.hpp"

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<sycl::int3>(1, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Test>(sycl::nd_range<1>(128, 8),
                                [=](sycl::nd_item<1>) {
                                  sycl::int3 vec1 = {1, 1, 1};
                                  sycl::int3 vec2 = {2, 2, 2};
                                  array[0] = vec1 / vec2;
                                });
   }).wait();
  // CHECK: WARNING: DeviceSanitizer: data race
  // CHECK-NEXT: When write of size 8 at 0x{{.*}} in kernel <{{.*}}Test>
  // CHECK-NEXT: #0 {{.*}}check_access16.cpp:[[@LINE-5]]

  return 0;
}
