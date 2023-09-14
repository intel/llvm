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

// CHECK-LABEL:   func.func @make() -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.null : !llvm.ptr
// CHECK-NEXT:      return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @makeF() -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.null : !llvm.ptr
// CHECK-NEXT:      %[[VAL_1:.*]] = "polygeist.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:      return %[[VAL_1]] : memref<?xf32>
// CHECK-NEXT:    }
