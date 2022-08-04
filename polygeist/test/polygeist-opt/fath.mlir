// RUN: polygeist-opt --loop-restructure --split-input-file %s | FileCheck %s

module {
  func.func @_Z14computeTempCPUPfS_iii(%arg2: i32, %arg3: i32, %arg4: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb1(%c0_i32, %c0_i32 : i32, i32)
  ^bb1(%0: i32, %1: i32):  // 3 preds: ^bb0, ^bb4, ^bb5
    %2 = arith.cmpi ult, %0, %arg3 : i32
    cf.cond_br %2, ^bb2, ^bb5
  ^bb2:
    %9 = arith.addi %0, %c1_i32 : i32
    cf.br ^bb1(%9, %1 : i32, i32)
  ^bb5:  // pred: ^bb1
    return
  }
}

// CHECK:   func.func @_Z14computeTempCPUPfS_iii(%arg0: i32, %arg1: i32, %arg2: i32) {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0:2 = scf.while (%arg3 = %c0_i32, %arg4 = %c0_i32) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:       %1 = arith.cmpi ult, %arg3, %arg1 : i32
// CHECK-NEXT:       %false = arith.constant false
// CHECK-NEXT:       %2:3 = scf.if %1 -> (i1, i32, i32) {
// CHECK-NEXT:         %3 = arith.addi %arg3, %c1_i32 : i32
// CHECK-NEXT:         %true = arith.constant true
// CHECK-NEXT:         scf.yield %true, %3, %arg4 : i1, i32, i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %false, %arg3, %arg4 : i1, i32, i32
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.condition(%2#0) %2#1, %2#2 : i32, i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg3: i32, %arg4: i32):
// CHECK-NEXT:       scf.yield %arg3, %arg4 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
