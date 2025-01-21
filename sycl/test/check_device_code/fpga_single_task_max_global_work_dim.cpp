// RUN: %clangxx -fsycl-device-only -fsycl-targets=spir64_fpga -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-FPGA
// RUN: %clangxx -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-DEFAULT

// Tests that single_task implicitly adds the max_global_work_dim when AOT
// compiling for FPGA.
// Additionally it checks that existing attributes do not cause conflicts.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  // CHECK-FPGA: spir_kernel void @_ZTSZ4mainE7Kernel1() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM:[0-9]+]]
  // CHECK-DEFAULT-NOT: spir_kernel void @_ZTSZ4mainE7Kernel1() {{.*}} !max_global_work_dim
  Q.single_task<class Kernel1>([]() {});
  // CHECK-FPGA: spir_kernel void @_ZTSZ4mainE7Kernel2() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM]]
  // CHECK-DEFAULT: spir_kernel void @_ZTSZ4mainE7Kernel2() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM:[0-9]+]]
  Q.single_task<class Kernel2>([]() [[intel::max_global_work_dim(0)]] {});
  // CHECK-FPGA: spir_kernel void @_ZTSZ4mainE7Kernel3() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM]]
  // CHECK-DEFAULT: spir_kernel void @_ZTSZ4mainE7Kernel3() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM]]
  Q.single_task<class Kernel3>(
      []() [[sycl::work_group_size_hint(1), intel::max_global_work_dim(0)]] {});
  // CHECK-FPGA: spir_kernel void @_ZTSZ4mainE7Kernel4() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM]]
  // CHECK-DEFAULT: spir_kernel void @_ZTSZ4mainE7Kernel4() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM]]
  Q.single_task<class Kernel4>(
      []() [[intel::max_global_work_dim(0), sycl::reqd_work_group_size(1)]] {});
  // CHECK-FPGA: spir_kernel void @_ZTSZ4mainE7Kernel5() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM]]
  // CHECK-DEFAULT: spir_kernel void @_ZTSZ4mainE7Kernel5() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM]]
  Q.single_task<class Kernel5>(
      []() [[sycl::work_group_size_hint(1), intel::max_global_work_dim(0),
             sycl::reqd_work_group_size(1)]] {});
  return 0;
}

// CHECK-FPGA: ![[MAX_GLOBAL_WORK_DIM:[0-9]+]] = !{i32 0}
