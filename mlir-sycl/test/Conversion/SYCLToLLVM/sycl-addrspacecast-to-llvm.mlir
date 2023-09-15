// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: llvm.func @PtrCastToGeneric(%arg0: !llvm.ptr) -> !llvm.ptr<4> {
// CHECK-NEXT:    %0 = llvm.addrspacecast %arg0 : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr<4>
// CHECK-NEXT:  }

func.func @PtrCastToGeneric(
    %arg0: memref<?xi32, #sycl.access.address_space<private>>)
    -> memref<?xi32, #sycl.access.address_space<generic>> {
  %0 = sycl.addrspacecast %arg0
      : memref<?xi32, #sycl.access.address_space<private>>
      to memref<?xi32, #sycl.access.address_space<generic>>
  return %0 : memref<?xi32, #sycl.access.address_space<generic>>
}

// -----

// CHECK-LABEL: llvm.func @GenericCastToPtr(%arg0: !llvm.ptr<4>) -> !llvm.ptr {
// CHECK-NEXT:    %0 = llvm.addrspacecast %arg0 : !llvm.ptr<4> to !llvm.ptr
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr
// CHECK-NEXT:  }

func.func @GenericCastToPtr(
    %arg0: memref<?xi32, #sycl.access.address_space<generic>>)
    -> memref<?xi32, #sycl.access.address_space<private>> {
  %0 = sycl.addrspacecast %arg0
      : memref<?xi32, #sycl.access.address_space<generic>>
      to memref<?xi32, #sycl.access.address_space<private>>
  return %0 : memref<?xi32, #sycl.access.address_space<private>>
}
