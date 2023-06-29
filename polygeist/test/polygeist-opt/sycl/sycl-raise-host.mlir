// RUN: polygeist-opt --sycl-raise-host --split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.module @device_functions
gpu.module @device_functions {
// CHECK:         gpu.func @foo() kernel
  gpu.func @foo() kernel {
    gpu.return
  }
}

// CHECK-LABEL: llvm.mlir.global private unnamed_addr constant @kernel_ref("foo\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
llvm.mlir.global private unnamed_addr constant @kernel_ref("foo\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}

// CHECK-LABEL: llvm.func @f() -> !llvm.ptr
// CHECK-NEXT:    %[[VAL_0:.*]] = sycl.host.get_kernel @device_functions::@foo : !llvm.ptr
// CHECK-NEXT:    llvm.return %[[VAL_0]] : !llvm.ptr
llvm.func @f() -> !llvm.ptr {
  %kn = llvm.mlir.addressof @kernel_ref : !llvm.ptr
  llvm.return %kn : !llvm.ptr
}
