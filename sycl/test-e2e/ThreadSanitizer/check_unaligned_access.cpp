// REQUIRES: linux, cpu || (gpu && level_zero)
// ALLOW_RETRIES: 10
// RUN: %{build} %device_tsan_flags -O2 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
#include "sycl/detail/core.hpp"
#include "sycl/usm.hpp"

struct __attribute__((packed)) S {
  char c;
  int x;
};

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<S>(1, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Test>(sycl::nd_range<1>(128, 8),
                                [=](sycl::nd_item<1>) { array[0].x++; });
   }).wait();
  // CHECK: WARNING: DeviceSanitizer: data race
  // CHECK-NEXT: When write of size 4 at 0x{{.*}} in kernel <{{.*}}Test>
  // CHECK-NEXT: #0 {{.*}}check_unaligned_access.cpp:[[@LINE-4]]

  sycl::free(array, Q);
  return 0;
}
