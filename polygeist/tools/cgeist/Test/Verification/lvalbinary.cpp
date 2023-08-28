// RUN: cgeist  %s --function=* -S -o - | FileCheck %s

// CHECK-LABEL:   func.func @_Z3fooRii(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32)
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           affine.store %[[VAL_1]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return
// CHECK:         }
void foo(int &x, int y) { (x = 0) = y; }

// CHECK-LABEL:   func.func @_Z3fooRiS_i(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<?xi32>, %[[VAL_1:.*]]: memref<?xi32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: i32) attributes
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           affine.store %[[VAL_3]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           affine.store %[[VAL_2]], %[[VAL_1]][0] : memref<?xi32>
// CHECK:           return
// CHECK:         }
void foo(int &x, int &y, int z) { (x = 0, y) = z; }
