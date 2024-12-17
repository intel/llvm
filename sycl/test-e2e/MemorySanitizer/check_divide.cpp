// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

// XFAIL: gpu-intel-gen12 || gpu-intel-dg2
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16184
// XFAIL: arch-intel_gpu_pvc
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16401

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;

  auto *array = sycl::malloc_device<int>(3, Q);

  Q.submit([&](sycl::handler &h) { h.single_task([=]() { array[1] = 1; }); });
  Q.wait();

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel1>([=]() { array[0] = array[0] / array[1]; });
  });
  Q.wait();
  // CHECK-NOT: kernel <{{.*MyKernel1}}>

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel2>([=]() { array[0] = array[0] / array[2]; });
  });
  Q.wait();
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel2}}>
  // CHECK: #0 {{.*}} {{.*check_divide.cpp}}:[[@LINE-5]]
  sycl::free(array, Q);

  return 0;
}
