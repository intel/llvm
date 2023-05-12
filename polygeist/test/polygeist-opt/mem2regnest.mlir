// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module {
  func.func @foo(%arg0: i1) -> i32 {
    %c-1_i32 = arith.constant -1 : i32
    %c20_i32 = arith.constant 20 : i32
    %c10_i32 = arith.constant 10 : i32
    %5 = memref.alloca() : memref<i32>
    %6 = llvm.mlir.undef : i32
    memref.store %6, %5[] : memref<i32>
        %a9 = scf.execute_region -> (i32) {
          cf.cond_br %arg0, ^bb1, ^bb3
        ^bb1:  // pred: ^bb0
          scf.yield %c20_i32 : i32
        ^bb3:  // 2 preds: ^bb0, ^bb2
          memref.store %c-1_i32, %5[] : memref<i32>
          scf.yield %c10_i32 : i32
        }
    scf.if %arg0 {
        memref.store %a9, %5[] : memref<i32>
    }
    %a10 = memref.load %5[] : memref<i32>
    return %a10 : i32
  }
}

// CHECK:   func.func @foo(%arg0: i1) -> i32 {
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c20_i32 = arith.constant 20 : i32
// CHECK-NEXT:     %c10_i32 = arith.constant 10 : i32
// CHECK-NEXT:     %0 = llvm.mlir.undef : i32
// CHECK-NEXT:     %1:2 = scf.execute_region -> (i32, i32) {
// CHECK-NEXT:       cf.cond_br %arg0, ^bb1, ^bb2
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       scf.yield %c20_i32, %0 : i32, i32
// CHECK-NEXT:     ^bb2:  // pred: ^bb0
// CHECK-NEXT:       scf.yield %c10_i32, %c-1_i32 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = scf.if %arg0 -> (i32) {
// CHECK-NEXT:       scf.yield %1#0 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %1#1 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2 : i32
// CHECK-NEXT:   }

