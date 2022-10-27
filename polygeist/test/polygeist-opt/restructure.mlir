// RUN: polygeist-opt --loop-restructure --split-input-file %s | FileCheck %s

module {
func.func @kernel_gemm(%arg0: i64) -> i1 {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  cf.br ^bb1(%c0_i64 : i64)
^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi "slt", %0, %c0_i64 : i64
  %5 = arith.cmpi "sle", %0, %arg0 : i64
  cf.cond_br %5, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %8 = arith.addi %0, %c1_i64 : i64
  cf.br ^bb1(%8 : i64)
^bb3:  // pred: ^bb1
  return %2 : i1
}


// CHECK:   func.func @kernel_gemm(%arg0: i64) -> i1 {
// CHECK-NEXT:     %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.mlir.undef : i1
// CHECK-NEXT:     %1:2 = scf.while (%arg1 = %c0_i64, %arg2 = %0) : (i64, i1) -> (i64, i1) {
// CHECK-NEXT:       %2 = arith.cmpi slt, %arg1, %c0_i64 : i64
// CHECK-NEXT:       %3 = arith.cmpi sle, %arg1, %arg0 : i64
// CHECK-NEXT:       %false = arith.constant false
// CHECK-NEXT:       %4:3 = scf.if %3 -> (i1, i64, i1) {
// CHECK-NEXT:         %5 = arith.addi %arg1, %c1_i64 : i64
// CHECK-NEXT:         %true = arith.constant true
// CHECK-NEXT:         scf.yield %true, %5, %2 : i1, i64, i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %false, %arg1, %2 : i1, i64, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.condition(%4#0) %4#1, %4#2 : i64, i1
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg1: i64, %arg2: i1):  
// CHECK-NEXT:       scf.yield %arg1, %arg2 : i64, i1
// CHECK-NEXT:     }
// CHECK-NEXT:     return %1#1 : i1
// CHECK-NEXT:   }


  func.func @gcd(%arg0: i32, %arg1: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = memref.alloca() : memref<i32>
    %1 = memref.alloca() : memref<i32>
    %2 = memref.alloca() : memref<i32>
    memref.store %arg0, %2[] : memref<i32>
    memref.store %arg1, %1[] : memref<i32>
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %3 = memref.load %1[] : memref<i32>
    %4 = arith.cmpi sgt, %3, %c0_i32 : i32
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = memref.load %0[] : memref<i32>
    %8 = memref.load %2[] : memref<i32>
    %9 = arith.remsi %8, %3 : i32
    scf.if %true {
      memref.store %9, %0[] : memref<i32>
    }
    memref.store %3, %2[] : memref<i32>
    memref.store %9, %1[] : memref<i32>
    cf.br ^bb1
  ^bb3:  // pred: ^bb1
    %7 = memref.load %2[] : memref<i32>
    return %7 : i32
  }


// CHECK:   func.func @gcd(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %true = arith.constant true
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     %alloca_1 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca_1[] : memref<i32>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:     scf.while : () -> () {
// CHECK-NEXT:       %1 = memref.load %alloca_0[] : memref<i32>
// CHECK-NEXT:       %2 = arith.cmpi sgt, %1, %c0_i32 : i32
// CHECK-NEXT:       %false = arith.constant false
// CHECK-NEXT:       %3 = scf.if %2 -> (i1) {
// CHECK-NEXT:         %4 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:         %5 = memref.load %alloca_1[] : memref<i32>
// CHECK-NEXT:         %6 = arith.remsi %5, %1 : i32
// CHECK-NEXT:         scf.if %true {
// CHECK-NEXT:           memref.store %6, %alloca[] : memref<i32>
// CHECK-NEXT:         }
// CHECK-NEXT:         memref.store %1, %alloca_1[] : memref<i32>
// CHECK-NEXT:         memref.store %6, %alloca_0[] : memref<i32>
// CHECK-NEXT:         %true_2 = arith.constant true
// CHECK-NEXT:         scf.yield %true_2 : i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %false : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.condition(%3)
// CHECK-NEXT:     } do {
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %0 = memref.load %alloca_1[] : memref<i32>
// CHECK-NEXT:     return %0 : i32
// CHECK-NEXT:   }

}
