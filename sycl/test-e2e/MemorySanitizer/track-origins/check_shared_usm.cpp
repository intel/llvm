// REQUIRES: linux, cpu || (gpu && level_zero)
// XFAIL: run-mode && linux && arch-intel_gpu_pvc && level_zero_v2_adapter
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19585

// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O0 -g -o %t1.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t1.out 2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-ORIGIN-STACK
// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O1 -g -o %t2.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O2 -g -o %t3.out
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:1 %{run} %t3.out 2>&1 | FileCheck %s
// RUN: env UR_LAYER_MSAN_OPTIONS=msan_check_host_and_shared_usm:0 %{run} %t3.out 2>&1 | FileCheck %s --check-prefixes CHECK-SHAREDUSM

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int data) { return data; }

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_shared<int>(2, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { array[0] = check(array[1]); });
  });
  Q.wait();
  // CHECK-NOT: [kernel]
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel}}>
  // CHECK: #{{.*}} {{.*check_shared_usm.cpp}}:[[@LINE-6]]
  // CHECK: ORIGIN: Shared USM allocation
  // CHECK-ORIGIN-STACK: #{{.*}} {{.*check_shared_usm.cpp}}:[[@LINE-11]]
  // CHECK-SHAREDUSM-NOT: use-of-uninitialized-value

  sycl::free(array, Q);
  return 0;
}
