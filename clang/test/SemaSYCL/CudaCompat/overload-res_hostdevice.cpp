// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify=host
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify=sycl-dev,all-dev
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fcuda-is-device"  "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++"  -verify=cuda-dev,all-dev

// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s -check-prefixes=CHECK,HOST
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s -check-prefixes=CHECK,DEVICE
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fcuda-is-device"  "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s -check-prefixes=CHECK,DEVICE

// Check that overload resolution favours the right overload based on its
// host/device/global attribute or raise the proper diagnostics.

#include "overloads.h"

// CHECK: FunctionDecl [[OVERLOAD_HOST_DEV:0x[^ ]*]] {{.*}} overload
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CUDAHostAttr
// CHECK-NEXT: CUDADeviceAttr
// CHECK: FunctionDecl [[OVERLOAD_DEV:0x[^ ]*]] {{.*}} overload
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CUDADeviceAttr
// CHECK: FunctionDecl [[OVERLOAD_HOST:0x[^ ]*]] {{.*}} overload
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CUDAHostAttr

__attribute__((host)) __attribute__((device)) void caller_hostdevice() {
    callee_device();
    callee_host(); // cuda-dev-error {{reference to __host__ function 'callee_host' in __host__ __device__ function}}
    // cuda-dev-note@#callee_host {{'callee_host' declared here}}
    callee_host_implicit(); // cuda-dev-error {{reference to __host__ function 'callee_host_implicit' in __host__ __device__ function}}
    // cuda-dev-note@#callee_host_implicit {{'callee_host_implicit' declared here}}
    callee_hostdevice();
    callee_hostdevice_implicit();
    callee_global(); // host-error {{call to global function 'callee_global' not configured}}
    // sycl-dev-error@-1 {{no matching function for call to 'callee_global'}}
    // sycl-dev-note@#callee_global {{candidate function not viable: call to __global__ function from __host__ __device__ function}}
    // cuda-dev-error@-3 {{reference to __global__ function 'callee_global' in __host__ __device__ function}}
    // cuda-dev-note@#callee_global {{'callee_global' declared here}}
}

__attribute__((host)) __attribute__((device)) void caller_hostdevice_overload_test() {
    // CHECK: FunctionDecl {{.*}} caller_hostdevice_overload_test
    // for host mode
    // HOST: DeclRefExpr {{.*}} Function [[OVERLOAD_HOST]]
    // for SYCL and CUDA device
    // DEVICE: DeclRefExpr {{.*}} Function [[OVERLOAD_DEV]]
    bar::overload();
}

#ifdef __SYCL_DEVICE_ONLY__
__attribute__((sycl_device)) void sycl_root() {
  caller_hostdevice();
}
#endif
