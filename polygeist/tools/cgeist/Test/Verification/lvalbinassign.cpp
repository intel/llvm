// RUN: cgeist --use-opaque-pointers %s --function=* -S -o - | FileCheck %s

// CHECK-LABEL:   func.func @_Z3fooRii(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           affine.store %[[VAL_1]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_1]] : i32
// CHECK:         }
int foo(int &x, int y) { return (x = 0) = y; }
