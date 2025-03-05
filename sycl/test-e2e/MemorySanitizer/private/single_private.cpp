// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O0 -g -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int p) { return p; }
__attribute__((noinline)) int foo(int *p) { return check(*p); }
// CHECK-NOT: [kernel]
// CHECK: DeviceSanitizer: use-of-uninitialized-value
// CHECK: #0 {{foo.*}} {{.*single_private.cpp}}:[[@LINE-3]]

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<int>(1, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() {
      int p[4];
      *array += foo(p);
    });
  });
  Q.wait();

  sycl::free(array, Q);
  return 0;
}
