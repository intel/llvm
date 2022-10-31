// RUN: cgeist -S --function=* %s | FileCheck %s

struct C {
  int a;
  double* b;
};

struct C* make() {
    return (struct C*)0;
}

float* makeF() {
    return (float*)0;
}

// CHECK:   func @make() -> !llvm.ptr<!llvm.struct<(i32, memref<?xf64>)>> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = llvm.mlir.null : !llvm.ptr<!llvm.struct<(i32, memref<?xf64>)>>
// CHECK-NEXT:     return %0 : !llvm.ptr<!llvm.struct<(i32, memref<?xf64>)>>
// CHECK-NEXT:   }
// CHECK:   func @makeF() -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<f32>) -> memref<?xf32>
// CHECK-NEXT:     return %1 : memref<?xf32>
// CHECK-NEXT:   }
