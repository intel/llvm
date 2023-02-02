// RUN: %clangxx -fsycl-device-only -fsycl-targets=spir64_fpga -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-FPGA
// RUN: %clangxx -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-DEFAULT

// Tests that single_task implicitly adds the max_global_work_dim when AOT
// compiling for FPGA.

// CHECK-DEFAULT-NOT: !max_global_work_dim

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  // CHECK-FPGA: spir_kernel void @_ZTSZ4mainE6Kernel() {{.*}} !max_global_work_dim ![[MAX_GLOBAL_WORK_DIM:[0-9]+]]
  Q.single_task<class Kernel>([]() {});
  return 0;
}

// CHECK-FPGA: ![[MAX_GLOBAL_WORK_DIM:[0-9]+]] = !{i32 0}
