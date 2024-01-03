// RUN: polygeist-opt --raise-scf-to-affine -allow-unregistered-dialect --split-input-file %s | FileCheck %s
// XFAIL: *

// COM: Expected to fail as `scf.parallel` is not in the correct shape and is not supported in this pass.

func.func @aff(%c : i1, %arg0: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.if %c {
    %75 = arith.index_cast %arg0 : i32 to index
    scf.parallel (%arg5) = (%c0) to (%75) step (%c1) {
      "test.op"() : () -> ()
  }
  return 
}

// CHECK-LABEL:   func.func @aff(
// CHECK-SAME:                   [[VAL0:%.*]]: i1,
// CHECK-SAME:                   [[VAL1:%.*]]: i32) {
// CHECK-NEXT:      [[VAL2:%.*]] = arith.index_cast [[VAL1]] : i32 to index
// CHECK-NEXT:      scf.if [[VAL0]] {
// CHECK-NEXT:        affine.parallel (%arg2) = (0) to (symbol([[VAL2]])) {
// CHECK-NEXT:          "test.op"() : () -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
