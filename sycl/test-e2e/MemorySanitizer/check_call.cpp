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
  auto *array = sycl::malloc_device<int>(2, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>(
        [=]() { array[0] = foo(array[0], array[1]); });
  });
  Q.wait();
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel}}>
  // CHECK: #0 {{.*}} {{.*check_call.cpp}}:[[@LINE-5]]

  sycl::free(array, Q);
  return 0;
}
