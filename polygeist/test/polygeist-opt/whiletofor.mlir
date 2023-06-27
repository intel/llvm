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

// -----

func.func private @foo(i64, i32)

// COM: This tests that we can convert an `scf.while` operation whose
// COM: `scf.condition` receives a value defined in the before block as an
// COM: argument to an `scf.for` operation.

// CHECK-LABEL:   func.func @extsi(
// CHECK-SAME:                     %[[VAL_0:.*]]: i64) -> i32 {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : index to i32
// CHECK-NEXT:      scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_2]] {
// CHECK-NEXT:        %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : index to i32
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
// CHECK-NEXT:        func.call @foo(%[[VAL_7]], %[[VAL_6]]) : (i64, i32) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[VAL_4]] : i32
// CHECK-NEXT:    }

func.func @extsi(%ub: i64) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %res:2 = scf.while (%arg = %c0_i32) : (i32) -> (i64, i32) {
    %ext = arith.extsi %arg : i32 to i64
    %cond = arith.cmpi slt, %ext, %ub : i64
    scf.condition(%cond) %ext, %arg : i64, i32
  } do {
   ^bb0(%arg0: i64, %arg1: i32):
    func.call @foo(%arg0, %arg1) : (i64, i32) -> ()
    %c1_i32 = arith.constant 1 : i32
    %next = arith.addi %arg1, %c1_i32 : i32
    scf.yield %next : i32
  }
  return %res#1 : i32
}
