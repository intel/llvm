// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice -emit-llvm -o - %s | FileCheck %s

#include "Inputs/sycl.hpp"
[[intel::scheduler_target_fmax_mhz(5)]] void
func() {}

template <int N>
[[intel::scheduler_target_fmax_mhz(N)]] void zoo() {}

int main() {
  cl::sycl::kernel_single_task<class test_kernel1>(
      []() [[intel::scheduler_target_fmax_mhz(2)]]{});

  cl::sycl::kernel_single_task<class test_kernel2>(
      []() { func(); });

  cl::sycl::kernel_single_task<class test_kernel3>(
      []() { zoo<75>(); });
}
// CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel1() {{.*}} !scheduler_target_fmax_mhz ![[PARAM1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel2() {{.*}} !scheduler_target_fmax_mhz ![[PARAM2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel3() {{.*}} !scheduler_target_fmax_mhz ![[PARAM3:[0-9]+]]
// CHECK: ![[PARAM1]] = !{i32 2}
// CHECK: ![[PARAM2]] = !{i32 5}
// CHECK: ![[PARAM3]] = !{i32 75}
