// RUN: polygeist-opt --loop-restructure --split-input-file %s | FileCheck %s

module {
  func.func @_Z14computeTempCPUPfS_iii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb1(%c0_i32, %c0_i32 : i32, i32)
  ^bb1(%0: i32, %1: i32):  // 3 preds: ^bb0, ^bb4, ^bb5
    %2 = arith.cmpi ult, %0, %arg3 : i32
    cf.cond_br %2, ^bb2(%c0_i32 : i32), ^bb5
  ^bb2(%3: i32):  // 2 preds: ^bb1, ^bb3
    %4 = arith.cmpi ult, %3, %arg2 : i32
    cf.cond_br %4, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %5 = arith.index_cast %3 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = memref.load %arg0[%6] : memref<?xf32>
    memref.store %7, %arg1[%5] : memref<?xf32>
    %8 = arith.addi %3, %c1_i32 : i32
    cf.br ^bb2(%8 : i32)
  ^bb4:  // pred: ^bb2
    %9 = arith.addi %0, %c1_i32 : i32
    cf.br ^bb1(%9, %1 : i32, i32)
  ^bb5:  // pred: ^bb1
    %10 = arith.addi %1, %c1_i32 : i32
    %11 = arith.cmpi ult, %10, %arg4 : i32
    cf.cond_br %11, ^bb1(%c0_i32, %10 : i32, i32), ^bb6
  ^bb6:  // pred: ^bb5
    return
  }
}

// CHECK:   func.func @_Z14computeTempCPUPfS_iii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0:2 = scf.while (%arg5 = %c0_i32, %arg6 = %c0_i32) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:       %1:3 = scf.execute_region -> (i1, i32, i32) {
// CHECK-NEXT:         %2 = arith.cmpi ult, %arg5, %arg3 : i32
// CHECK-NEXT:         cf.cond_br %2, ^bb3(%c0_i32 : i32), ^bb1
// CHECK-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-NEXT:         %3 = arith.addi %arg6, %c1_i32 : i32
// CHECK-NEXT:         %4 = arith.cmpi ult, %3, %arg4 : i32
// CHECK-NEXT:         %false = arith.constant false
// CHECK-NEXT:         %true = arith.constant true
// CHECK-NEXT:         cf.cond_br %4, ^bb2(%true, %c0_i32, %3 : i1, i32, i32), ^bb2(%false, %arg5, %arg6 : i1, i32, i32)
// CHECK-NEXT:       ^bb2(%5: i1, %6: i32, %7: i32):  // 3 preds: ^bb1, ^bb1, ^bb3
// CHECK-NEXT:         scf.yield %5, %6, %7 : i1, i32, i32
// CHECK-NEXT:       ^bb3(%8: i32):  // pred: ^bb0
// CHECK-NEXT:         %9 = scf.while (%arg7 = %8) : (i32) -> i32 {
// CHECK-NEXT:           %11 = arith.cmpi ult, %arg7, %arg2 : i32
// CHECK-NEXT:           %false_1 = arith.constant false
// CHECK-NEXT:           %12:2 = scf.if %11 -> (i1, i32) {
// CHECK-NEXT:             %13 = arith.index_cast %arg7 : i32 to index
// CHECK-NEXT:             %14 = arith.index_cast %arg7 : i32 to index
// CHECK-NEXT:             %15 = memref.load %arg0[%14] : memref<?xf32>
// CHECK-NEXT:             memref.store %15, %arg1[%13] : memref<?xf32>
// CHECK-NEXT:             %16 = arith.addi %arg7, %c1_i32 : i32
// CHECK-NEXT:             %true_2 = arith.constant true
// CHECK-NEXT:             scf.yield %true_2, %16 : i1, i32
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %false_1, %arg7 : i1, i32
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.condition(%12#0) %12#1 : i32
// CHECK-NEXT:         } do {
// CHECK-NEXT:         ^bb0(%arg7: i32):
// CHECK-NEXT:           scf.yield %arg7 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         %10 = arith.addi %arg5, %c1_i32 : i32
// CHECK-NEXT:         %true_0 = arith.constant true
// CHECK-NEXT:         cf.br ^bb2(%true_0, %10, %arg6 : i1, i32, i32)
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.condition(%1#0) %1#1, %1#2 : i32, i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg5: i32, %arg6: i32):  
// CHECK-NEXT:       scf.yield %arg5, %arg6 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
