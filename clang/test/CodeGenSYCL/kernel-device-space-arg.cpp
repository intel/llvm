// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -disable-llvm-passes -o - | FileCheck %s

// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE15kernel_function(i32 addrspace(5)* {{.*}} i32 addrspace(6)* {{.*}}

#include "Inputs/sycl.hpp"

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
