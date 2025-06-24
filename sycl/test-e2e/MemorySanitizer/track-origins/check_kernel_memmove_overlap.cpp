// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O2 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) char check(char data1) { return data1; }

void overlap() {
  sycl::queue Q;
  constexpr size_t N = 1024;
  auto *array = sycl::malloc_shared<char>(N, Q);

  Q.submit([&](sycl::handler &h) {
     h.single_task<class MyKernel1>([=]() { memset(array, 0, N / 2); });
   }).wait();

  Q.submit([&](sycl::handler &h) {
     h.single_task<class MyKernel2>([=]() {
       check(array[0]);
       check(array[1]);
       memmove(array, array + N / 2 - 1, N / 2);
       check(array[0]);
       check(array[1]);
     });
   }).wait();
  // CHECK-NOT: [kernel]
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel2}}>
  // CHECK: #{{.*}} {{.*check_kernel_memmove_overlap.cpp}}:[[@LINE-6]]
  // CHECK: ORIGIN: Shared USM allocation
  // CHECK: #{{.*}} {{.*check_kernel_memmove_overlap.cpp}}:[[@LINE-20]]

  sycl::free(array, Q);
}

int main() {
  overlap();
  return 0;
}
