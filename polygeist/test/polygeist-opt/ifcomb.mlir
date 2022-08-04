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

// CHECK:   func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: memref<f32>, %arg1: i32, %arg2: i32, %arg3: i32) -> i8 {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = arith.cmpi sge, %arg3, %arg1 : i32
// CHECK-NEXT:     %1 = arith.cmpi sle, %arg3, %arg2 : i32
// CHECK-NEXT:     %2 = arith.andi %0, %1 : i1
// CHECK-NEXT:     %3 = arith.andi %0, %1 : i1
// CHECK-NEXT:     %4 = arith.extui %3 : i1 to i8
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       affine.store %cst, %arg0[] : memref<f32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %4 : i8
// CHECK-NEXT:   }
