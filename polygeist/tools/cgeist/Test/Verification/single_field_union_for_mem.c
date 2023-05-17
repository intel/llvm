// RUN: cgeist --use-opaque-pointers %s -O0 --function=* -S | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: cgeist --use-opaque-pointers %s -O0 --function=* -S -emit-llvm | FileCheck %s --check-prefix=CHECK-LLVM

struct foo {
  union int_wrapper {
    int *ptr;
  } d;
};

// CHECK-MLIR-LABEL:   func.func @id(
// CHECK-MLIR-SAME:                  %[[VAL_0:.*]]: !llvm.struct<(!llvm.struct<(memref<?xi32>)>)>) -> !llvm.struct<(!llvm.struct<(memref<?xi32>)>)>
// CHECK-MLIR-NEXT:      return %[[VAL_0]] : !llvm.struct<(!llvm.struct<(memref<?xi32>)>)>
// CHECK-MLIR-NEXT:    }

// CHECK-LLVM-LABEL:   define { { ptr } } @id(
// CHECK-LLVM-SAME:                            { { ptr } } %[[VAL_0:.*]]) {
// CHECK-LLVM-NEXT:      ret { { ptr } } %[[VAL_0]]
// CHECK-LLVM-NEXT:    }

struct foo id(struct foo f) {
  return f;
}
