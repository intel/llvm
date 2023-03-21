// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

module {
  func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: memref<f32>, %arg1: i32, %arg2: i32, %arg3: i32) -> i8 {
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.cmpi sge, %arg3, %arg1 : i32
    %1 = scf.if %0 -> (i8) {
      %2 = arith.cmpi sle, %arg3, %arg2 : i32
      %3 = scf.if %2 -> (i8) {
        affine.store %cst, %arg0[] : memref<f32>
        scf.yield %c1_i8 : i8
      } else {
        scf.yield %c0_i8 : i8
      }
      scf.yield %3 : i8
    } else {
      scf.yield %c0_i8 : i8
    }
    return %1 : i8
  }
}

// CHECK-LABEL:   func.func @_Z17compute_tran_tempPfPS_iiiiiiii(
// CHECK-SAME:          %[[VAL_0:.*]]: memref<f32>,
// CHECK-SAME:          %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32,
// CHECK-SAME:          %[[VAL_3:.*]]: i32) -> i8 {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = arith.cmpi sge, %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_7:.*]] = scf.if %[[VAL_6]] -> (i8) {
// CHECK:             %[[VAL_8:.*]] = arith.cmpi sle, %[[VAL_3]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.extui %[[VAL_8]] : i1 to i8
// CHECK:             scf.if %[[VAL_8]] {
// CHECK:               affine.store %[[VAL_5]], %[[VAL_0]][] : memref<f32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_9]] : i8
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_4]] : i8
// CHECK:           }
// CHECK:           return %[[VAL_10:.*]] : i8
// CHECK:         }

// FIXME: This is the output if some canonicalization patterns defined in the
//        polygeist dialect are applied. See comment in Ops.cpp.
//
// COM:   func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: memref<f32>, %arg1: i32, %arg2: i32, %arg3: i32) -> i8 {
// COM-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// COM-NEXT:     %0 = arith.cmpi sge, %arg3, %arg1 : i32
// COM-NEXT:     %1 = arith.cmpi sle, %arg3, %arg2 : i32
// COM-NEXT:     %2 = arith.andi %0, %1 : i1
// COM-NEXT:     %3 = arith.extui %2 : i1 to i8
// COM-NEXT:     scf.if %2 {
// COM-NEXT:       affine.store %cst, %arg0[] : memref<f32>
// COM-NEXT:     }
// COM-NEXT:     return %3 : i8
// COM-NEXT:   }
