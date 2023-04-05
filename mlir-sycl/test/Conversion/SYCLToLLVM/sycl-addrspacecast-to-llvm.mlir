// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm='use-opaque-pointers=1' -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: llvm.func @PtrCastToGeneric(%arg0: !llvm.ptr) -> !llvm.ptr<4> {
// CHECK-NEXT:    %0 = llvm.addrspacecast %arg0 : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr<4>
// CHECK-NEXT:  }

func.func @PtrCastToGeneric(%arg0: memref<?xi32>) -> memref<?xi32, 4> {
  %0 = sycl.addrspacecast %arg0 : memref<?xi32> to memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// -----

// CHECK-LABEL: llvm.func @GenericCastToPtr(%arg0: !llvm.ptr<4>) -> !llvm.ptr {
// CHECK-NEXT:    %0 = llvm.addrspacecast %arg0 : !llvm.ptr<4> to !llvm.ptr
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr
// CHECK-NEXT:  }

func.func @GenericCastToPtr(%arg0: memref<?xi32, 4>) -> memref<?xi32> {
  %0 = sycl.addrspacecast %arg0 : memref<?xi32, 4> to memref<?xi32>
  return %0 : memref<?xi32>
}
