// REQUIRES: linux, cpu || (gpu && level_zero)
// XFAIL: run-mode && linux && arch-intel_gpu_pvc && level_zero_v2_adapter
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19585

// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O0 -g -o %t0.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t0.out 2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-ORIGIN-STACK
// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O2 -g -o %t1.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t1.out 2>&1 | FileCheck %s

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
  // CHECK-ORIGIN-STACK: #{{.*}} {{.*check_kernel_memmove_overlap.cpp}}:[[@LINE-20]]

  sycl::free(array, Q);
}

int main() {
  overlap();
  return 0;
}
