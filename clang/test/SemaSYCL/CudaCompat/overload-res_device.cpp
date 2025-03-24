// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify -DDEVICE
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify -DDEVICE
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fcuda-is-device"  "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify -DDEVICE
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fcuda-is-device"  "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify

// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump -DDEVICE 2> /dev/null | FileCheck %s
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump -DDEVICE 2> /dev/null | FileCheck %s
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fcuda-is-device"  "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump -DDEVICE 2> /dev/null | FileCheck %s
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fcuda-is-device"  "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s

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

#ifdef DEVICE
#define DEVICE_ATTR __attribute__((device))
#else
#define DEVICE_ATTR __attribute__((global))
#endif

DEVICE_ATTR void caller_device() {
    callee_device();
    callee_host(); // expected-error {{no matching function for call to 'callee_host'}}
    #ifdef DEVICE
    // expected-note@#callee_host {{candidate function not viable: call to __host__ function from __device__ function}}
    #else
    // expected-note@#callee_host {{candidate function not viable: call to __host__ function from __global__ function}}
    #endif
    callee_host_implicit(); // expected-error {{no matching function for call to 'callee_host_implicit'}}
    #ifdef DEVICE
    // expected-note@#callee_host_implicit {{candidate function not viable: call to __host__ function from __device__ function}}
    #else
    // expected-note@#callee_host_implicit {{candidate function not viable: call to __host__ function from __global__ function}}
    #endif
    callee_hostdevice();
    callee_hostdevice_implicit();
    callee_global(); // expected-error {{no matching function for call to 'callee_global'}}
    #ifdef DEVICE
    // expected-note@#callee_global {{candidate function not viable: call to __global__ function from __device__ function}}
    #else
    // expected-note@#callee_global {{candidate function not viable: call to __global__ function from __global__ function}}
    #endif
}

DEVICE_ATTR void caller_device_overload_test() {
  // CHECK: FunctionDecl {{.*}} caller_device_overload_test
  // CHECK: DeclRefExpr {{.*}} Function [[OVERLOAD_DEV]]
  bar::overload();
}

#ifdef __SYCL_DEVICE_ONLY__
#ifdef DEVICE
__attribute__((sycl_device)) void sycl_root() { caller_device(); }
#endif
#endif
