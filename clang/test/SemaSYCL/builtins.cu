// RUN: %clang_cc1 \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -sycl-std=2020 -ast-dump %s | FileCheck %s

// Check that aux builtins have the correct device attribute (i.e., __device__).

__attribute__((device)) void df() {
  int x = __nvvm_read_ptx_sreg_ctaid_x();
}

void fun() {
  df();
}

// CHECK: FunctionDecl {{.*}} __nvvm_read_ptx_sreg_ctaid_x
// CHECK-NEXT: BuiltinAttr
// CHECK-NEXT: NoThrowAttr
// CHECK-NEXT: ConstAttr
// CHECK-NEXT: CUDADeviceAttr
