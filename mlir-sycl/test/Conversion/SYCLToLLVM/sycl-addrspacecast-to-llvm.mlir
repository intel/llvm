// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm="use-bare-ptr-call-conv" -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: llvm.func @addrspacecast(%arg0: !llvm.ptr<i32>) -> !llvm.ptr<i32, 4> {
// CHECK-NEXT:    %0 = llvm.addrspacecast %arg0 : !llvm.ptr<i32> to !llvm.ptr<i32, 4>
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr<i32, 4>
// CHECK-NEXT:  }

func.func @addrspacecast(%arg0: memref<?xi32>) -> memref<?xi32, 4> {
  %0 = "sycl.addrspacecast"(%arg0) : (memref<?xi32>) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}
