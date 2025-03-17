// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_tsan_flags -DMALLOC_DEVICE -O0 -g -o %t1.out
// RUN: %{run} %deflake %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_tsan_flags -DMALLOC_DEVICE -O2 -g -o %t2.out
// RUN: %{run} %deflake %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_tsan_flags -DMALLOC_HOST -O2 -g -o %t3.out
// RUN: %{run} %deflake %t3.out 2>&1 | FileCheck %s
// RUN: %{build} %device_tsan_flags -DMALLOC_SHARED -O2 -g -o %t4.out
// RUN: %{run} %deflake %t4.out 2>&1 | FileCheck %s
#include "sycl/detail/core.hpp"
#include "sycl/usm.hpp"

int main() {
  sycl::queue Q;
#if defined(MALLOC_DEVICE)
  auto *array = sycl::malloc_device<char>(1, Q);
#elif defined(MALLOC_HOST)
  auto *array = sycl::malloc_host<char>(1, Q);
#else // defined(MALLOC_SHARED)
  auto *array = sycl::malloc_shared<char>(1, Q);
#endif

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Test>(sycl::nd_range<1>(32, 8),
                                [=](sycl::nd_item<1>) { array[0]++; });
   }).wait();
  // CHECK: WARNING: DeviceSanitizer: data race
  // CHECK-NEXT: When write of size 1 at 0x{{.*}} in kernel <{{.*}}Test>
  // CHECK-NEXT: #0 {{.*}}check_usm.cpp:[[@LINE-4]]

  sycl::free(array, Q);
  return 0;
}
