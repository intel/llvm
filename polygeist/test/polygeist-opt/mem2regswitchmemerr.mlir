// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module {
  func.func @parse(%arg0: i32) {
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %0 = llvm.mlir.undef : i32
    %1 = memref.alloca() : memref<i1>
    memref.store %true, %1[] : memref<i1>
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb5
    cf.cond_br %true, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %2 = memref.load %1[] : memref<i1>
    cf.switch %arg0 : i32, [
      default: ^bb5,
      104: ^bb3,
      113: ^bb4(%0 : i32)
    ]
  ^bb3:  // pred: ^bb2
    %3 = scf.if %true -> (i32) {
      memref.store %false, %1[] : memref<i1>
      scf.yield %c1_i32 : i32
    } else {
      scf.yield %0 : i32
    }
    cf.br ^bb4(%3 : i32)
  ^bb4(%4: i32):  // 2 preds: ^bb2, ^bb3
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb2, ^bb4
    cf.br ^bb1
  ^bb6:  // pred: ^bb1
    return
  }
}

// CHECK:   func.func @parse(%arg0: i32) {
// CHECK-DAG:     %false = arith.constant false
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %true = arith.constant true
// CHECK-DAG:     %0 = llvm.mlir.undef : i32
// CHECK-NEXT:     cf.br ^bb1(%true : i1)
// CHECK-NEXT:   ^bb1(%1: i1):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     cf.cond_br %true, ^bb2, ^bb6
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     cf.switch %arg0 : i32, [
// CHECK-NEXT:       default: ^bb5(%1 : i1),
// CHECK-NEXT:       104: ^bb3,
// CHECK-NEXT:       113: ^bb4(%0, %1 : i32, i1)
// CHECK-NEXT:     ]
// CHECK-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-NEXT:     %2:2 = scf.if %true -> (i32, i1) {
// CHECK-NEXT:       scf.yield %c1_i32, %false : i32, i1
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %0, %1 : i32, i1
// CHECK-NEXT:     }
// CHECK-NEXT:     cf.br ^bb4(%2#0, %2#1 : i32, i1)
// CHECK-NEXT:   ^bb4(%3: i32, %4: i1):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:     cf.br ^bb5(%4 : i1)
// CHECK-NEXT:   ^bb5(%5: i1):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT:     cf.br ^bb1(%5 : i1)
// CHECK-NEXT:   ^bb6:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   }
