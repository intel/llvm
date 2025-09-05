// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O0 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

// XFAIL: spirv-backend && gpu && run-mode
// XFAIL-TRACKER: https://github.com/llvm/llvm-project/issues/122075

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int data) { return data; }

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<int>(2, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(sycl::range<1>(2), [=](sycl::id<1> id) {
      array[id] = check(array[0]);
    });
  });
  Q.wait();
  // CHECK-NOT: [kernel]
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel}}>
  // CHECK: #0 {{.*}} {{.*check_no_local_size.cpp}}:[[@LINE-7]]

  sycl::free(array, Q);
  return 0;
}
