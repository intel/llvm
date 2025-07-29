// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify=all
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify=all
// RUN: %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fcuda-is-device"  "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" %t "-x" "c++" -verify=all

// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s -check-prefixes=CHECK,HOST
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s -check-prefixes=CHECK,SYCL-DEV
// RUN: not %clang_cc1 %s "-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-host" "-fcuda-is-device"  "-fsycl-cuda-compatibility" "-aux-triple" "nvptx64-nvidia-cuda" "-o" - "-x" "c++" -ast-dump 2> /dev/null | FileCheck %s -check-prefixes=CHECK,HOST

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

__attribute__((host)) void caller_host() {
    // FIXME: we probably need something better. In CUDA device mode, this should be an error.
    callee_device();
    callee_host();
    callee_host_implicit();
    callee_hostdevice();
    callee_hostdevice_implicit();
    callee_global(); // all-error {{call to global function 'callee_global' not configured}}
}

__attribute__((host)) void caller_host_overload_test() {
  // CHECK: FunctionDecl {{.*}} caller_host_overload_test
  // for host mode and cuda device
  // HOST: DeclRefExpr {{.*}} Function [[OVERLOAD_HOST]]
  // for SYCL device
  // SYCL-DEV: DeclRefExpr {{.*}} Function [[OVERLOAD_DEV]]
  bar::overload();
}
