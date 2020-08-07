// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -disable-llvm-passes -o - | FileCheck %s

// Pointers are wrapped inside structs.
// CHECK: %[[WRAPPER_GLOB_DEV:[a-zA-Z0-9_.]+]] = type { i32 addrspace(5)* }
// CHECK: %[[WRAPPER_GLOB_HOST:[a-zA-Z0-9_.]+]] = type { i32 addrspace(6)* }

// CHECK: define spir_kernel void {{.*}}kernel_function
// CHECK-SAME: %[[WRAPPER_GLOB_DEV]]* byval(%[[WRAPPER_GLOB_DEV]])
// CHECK-SAME: %[[WRAPPER_GLOB_HOST]]* byval(%[[WRAPPER_GLOB_HOST]])

#include "sycl.hpp"

int main() {
  __attribute__((opencl_global_device)) int *GLOBDEV = nullptr;
  __attribute__((opencl_global_host)) int *GLOBHOST = nullptr;
  cl::sycl::kernel_single_task<class kernel_function>(
      [=]() {
        __attribute__((opencl_global_device)) int *DevPtr = GLOBDEV;
        __attribute__((opencl_global_host)) int *HostPtr = GLOBHOST;
      });
  return 0;
}
