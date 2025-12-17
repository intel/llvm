// REQUIRES: linux, cpu || (gpu && level_zero)
// XFAIL: run-mode && linux && arch-intel_gpu_pvc && level_zero_v2_adapter
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19585

// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O0 -g -o %t0.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t0.out 2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-ORIGIN-STACK
// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O2 -g -o %t1.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t1.out 2>&1 | FileCheck %s
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:0 %{run} %t1.out 2>&1 | FileCheck %s --check-prefixes CHECK-HOSTUSM

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int data) { return data; }

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_host<int>(2, Q);
  array[0] = array[1] = 0;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { array[0] = check(array[1]); });
  });
  Q.wait();
  // CHECK-NOT: [kernel]
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel}}>
  // CHECK: #{{.*}} {{.*check_host_usm_initialized_on_host.cpp}}:[[@LINE-6]]
  // CHECK: ORIGIN: Host USM allocation
  // CHECK-ORIGIN-STACK: #{{.*}} {{.*check_host_usm_initialized_on_host.cpp}}:[[@LINE-12]]
  // CHECK-HOSTUSM-NOT: use-of-uninitialized-value

  sycl::free(array, Q);
  return 0;
}
