// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module {
  func.func @kernel_correlation(%arg0: memref<?xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %2 = memref.alloca() : memref<i1>
    memref.store %true, %2[] : memref<i1>
    cf.br ^bb1(%c0 : index)
  ^bb1(%3 : index):  // 2 preds: ^bb0, ^bb2
    %4 = arith.cmpi slt, %3, %c10 : index
    %5 = memref.load %2[] : memref<i1>
    %6 = arith.andi %4, %5 : i1
    cf.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    memref.store %cst, %arg0[%3] : memref<?xf64>
    %10 = arith.addi %3, %c1 : index
    cf.br ^bb1(%10 : index)
  ^bb3:  // pred: ^bb1
    return
  }
}

// CHECK:   func.func @kernel_correlation(%arg0: memref<?xf64>) {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c10 = arith.constant 10 : index
// CHECK-NEXT:     cf.br ^bb1(%c0 : index)
// CHECK-NEXT:   ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %1 = arith.cmpi slt, %0, %c10 : index
// CHECK-NEXT:     %2 = arith.andi %1, %true : i1
// CHECK-NEXT:     cf.cond_br %2, ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     memref.store %cst, %arg0[%0] : memref<?xf64>
// CHECK-NEXT:     %3 = arith.addi %0, %c1 : index
// CHECK-NEXT:     cf.br ^bb1(%3 : index)
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   }

