// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module  {
  func.func @bad(%arg0: i1, %arg1: i1, %arg2: memref<?xi32>) -> i64 {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = memref.alloca() : memref<i64>
    memref.store %c0_i64, %0[] : memref<i64>
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    scf.if %arg1 {
    } else {
      scf.execute_region {
        cf.cond_br %arg0, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        memref.store %c1_i64, %0[] : memref<i64>
        scf.yield
      ^bb3:  // pred: ^bb1
        scf.yield
      }
    }
    cf.br ^bb3
  ^bb3:  // pred: ^bb1
        %8 = memref.load %0[] : memref<i64>
    return %8 : i64
  }
}

// CHECK:   func.func @bad(%arg0: i1, %arg1: i1, %arg2: memref<?xi32>) -> i64 {
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:     cf.br ^bb1
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     %0 = scf.if %arg1 -> (i64) {
// CHECK-NEXT:       scf.yield %c0_i64 : i64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %1 = scf.execute_region -> i64 {
// CHECK-NEXT:         cf.cond_br %arg0, ^bb1, ^bb2
// CHECK-NEXT:       ^bb1:  // pred: ^bb0
// CHECK-NEXT:         scf.yield %c1_i64 : i64
// CHECK-NEXT:       ^bb2:  // pred: ^bb0
// CHECK-NEXT:         scf.yield %c0_i64 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %1 : i64
// CHECK-NEXT:     }
// CHECK-NEXT:     cf.br ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return %0 : i64
// CHECK-NEXT:   }
