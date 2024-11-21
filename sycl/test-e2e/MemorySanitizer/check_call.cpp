// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) long long foo(int data1, long long data2) {
  return data1 + data2;
}

int main() {
  sycl::queue Q;
  auto *array1 = sycl::malloc_device<int>(2, Q);
  auto *array2 = sycl::malloc_device<long long>(2, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>(
        [=]() { array2[0] = foo(array1[0], array2[1]); });
  });
  Q.wait();
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel}}>
  // CHECK: #0 {{.*}} {{.*check_call.cpp}}:[[@LINE-5]]

  sycl::free(array1, Q);
  sycl::free(array2, Q);
  return 0;
}
