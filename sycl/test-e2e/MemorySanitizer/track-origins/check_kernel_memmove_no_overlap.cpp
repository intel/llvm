// REQUIRES: linux, cpu || (gpu && level_zero)
// XFAIL: run-mode && arch-intel_gpu_pvc && !igc-dev
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19585

// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O0 -g -o %t0.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t0.out 2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-ORIGIN-STACK
// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O2 -g -o %t1.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -DINIT_SOURCE -O2 -g -o %t2.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t2.out 2>&1 | FileCheck %s --check-prefixes CHECK-INIT

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) char check(char data1) { return data1; }

void no_overlap() {
  sycl::queue Q;
  constexpr size_t N = 1024;
  auto *array1 = sycl::malloc_host<char>(N, Q);
  auto *array2 = sycl::malloc_host<char>(N, Q);

#ifdef INIT_SOURCE
  Q.submit([&](sycl::handler &h) {
     h.single_task<class MyKernel1>([=]() { memset(array1, 0, N); });
   }).wait();
#endif

  Q.submit([&](sycl::handler &h) {
     h.single_task<class MyKernel2>([=]() { memset(array2, 0, N); });
   }).wait();

  Q.submit([&](sycl::handler &h) {
     h.single_task<class MyKernel3>([=]() {
       memmove(array2, array1, N);
       check(array2[0]);
     });
   }).wait();
  // CHECK-NOT: [kernel]
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel3}}>
  // CHECK: #{{.*}} {{.*check_kernel_memmove_no_overlap.cpp}}:[[@LINE-6]]
  // CHECK: ORIGIN: Host USM allocation
  // CHECK-ORIGIN-STACK: #{{.*}} {{.*check_kernel_memmove_no_overlap.cpp}}:[[@LINE-24]]
  // CHECK-INIT-NOT: use-of-uninitialized-value

  sycl::free(array1, Q);
  sycl::free(array2, Q);
}

int main() {
  no_overlap();
  return 0;
}
