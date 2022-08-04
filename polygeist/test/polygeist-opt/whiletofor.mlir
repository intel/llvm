// RUN: polygeist-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
  func.func @set(%arg0: memref<?xi32>, %arg1: i64) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %1 = arith.extsi %arg2 : i32 to i64
      %2 = arith.cmpi slt, %1, %arg1 : i64
      scf.condition(%2) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):  // no predecessors
      %1 = arith.index_cast %arg2 : i32 to index
      memref.store %c0_i32, %arg0[%1] : memref<?xi32>
      %2 = arith.addi %arg2, %c1_i32 : i32
      scf.yield %2 : i32
    }
    return
  }
}

// CHECK:   func.func @set(%arg0: memref<?xi32>, %arg1: i64) {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i64 to index
// CHECK-NEXT:     scf.for %arg2 = %c0 to %0 step %c1 {
// CHECK-NEXT:       %1 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:       %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:       memref.store %c0_i32, %arg0[%2] : memref<?xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
