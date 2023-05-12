// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module {
  func.func @matrix_power(%arg0: memref<20xi32>, %arg1: i1, %arg2: index) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %0 = memref.alloca() : memref<index>
    memref.store %arg2, %0[] : memref<index>
    cf.br ^bb02
  ^bb02:  // pred: ^bb1
      scf.if %arg1 {
        memref.store %c0, %0[] : memref<index>
      }
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      %7 = memref.load %0[] : memref<index>
      %8 = arith.cmpi slt, %7, %c20 : index
      cf.cond_br %8, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      memref.store %c0_i32, %arg0[%7] : memref<20xi32>
      %24 = arith.addi %7, %c1 : index
      memref.store %24, %0[] : memref<index>
      cf.br ^bb1
    ^bb3:  // pred: ^bb1
      return
  }
}

// CHECK:   func.func @matrix_power(%arg0: memref<20xi32>, %arg1: i1, %arg2: index) {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c20 = arith.constant 20 : index
// CHECK-NEXT:     cf.br ^bb1
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     %0 = scf.if %arg1 -> (index) {
// CHECK-NEXT:       scf.yield %c0 : index
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %arg2 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     cf.br ^bb2(%0 : index)
// CHECK-NEXT:   ^bb2(%1: index):  // 2 preds: ^bb1, ^bb3
// CHECK-NEXT:     %2 = arith.cmpi slt, %1, %c20 : index
// CHECK-NEXT:     cf.cond_br %2, ^bb3, ^bb4
// CHECK-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-NEXT:     memref.store %c0_i32, %arg0[%1] : memref<20xi32>
// CHECK-NEXT:     %3 = arith.addi %1, %c1 : index
// CHECK-NEXT:     cf.br ^bb2(%3 : index)
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     return
// CHECK-NEXT:   }
