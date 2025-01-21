// RUN: %clang_cc1 \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -sycl-std=2020 -ast-dump %s | FileCheck %s

// Check that aux builtins have the correct device attribute.

#include "../CodeGenCUDA/Inputs/cuda.h"

__attribute__((device)) void df_0() {
  int x = __nvvm_read_ptx_sreg_ctaid_x();
}

__device__ void df_1() {
  int y = __nvvm_read_ptx_sreg_ctaid_y();
}

void fun() {
  df_0();
  df_1();
}

// CHECK: FunctionDecl {{.*}} __nvvm_read_ptx_sreg_ctaid_x
// CHECK-NEXT: BuiltinAttr
// CHECK-NEXT: NoThrowAttr
// CHECK-NEXT: ConstAttr
// CHECK-NEXT: CUDADeviceAttr

// CHECK: FunctionDecl {{.*}} __nvvm_read_ptx_sreg_ctaid_y
// CHECK-NEXT: BuiltinAttr
// CHECK-NEXT: NoThrowAttr
// CHECK-NEXT: ConstAttr
// CHECK-NEXT: CUDADeviceAttr
