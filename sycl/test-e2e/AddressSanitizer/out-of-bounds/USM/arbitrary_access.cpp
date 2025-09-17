// REQUIRES: linux, gpu && level_zero
// RUN: %{build} %device_asan_flags -Xarch_device -mllvm=-asan-spir-shadow-bounds=1 -O0 -g -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -Xarch_device -mllvm=-asan-spir-shadow-bounds=1 -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

void out_of_bounds_function() { *(int *)0xdeadbeef = 42; }
// CHECK: out-of-bounds-access
// CHECK-SAME: 0xdeadbeef
// CHECK: WRITE of size 4 at kernel {{<.*MyKernel>}}
// CHECK: {{.*arbitrary_access.cpp}}:[[@LINE-4]]

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { out_of_bounds_function(); });
  });
  Q.wait();

  return 0;
}
