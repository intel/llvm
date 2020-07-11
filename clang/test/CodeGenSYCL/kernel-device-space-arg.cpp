// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s

// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE15kernel_function(i32 addrspace(5)* {{.*}} i32 addrspace(6)* {{.*}}

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
