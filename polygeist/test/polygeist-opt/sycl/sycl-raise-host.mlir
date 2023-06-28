// RUN: polygeist-opt --sycl-raise-host --split-input-file %s | FileCheck %s

gpu.module @device_functions {
  gpu.func @foo() kernel {
    gpu.return
  }
}

// CHECK-LABEL: sycl.host.kernel_name @kernel_ref -> @device_functions::@foo
llvm.mlir.global private unnamed_addr constant @kernel_ref("foo\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}

llvm.func @f() -> !llvm.ptr {
  // CHECK-LABEL: %0 = sycl.host.get_kernel @device_functions::@foo : !llvm.ptr
  %kn = llvm.mlir.addressof @kernel_ref : !llvm.ptr
  llvm.return %kn : !llvm.ptr
}
