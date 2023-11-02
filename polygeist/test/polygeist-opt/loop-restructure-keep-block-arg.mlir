// RUN: polygeist-opt --allow-unregistered-dialect --loop-restructure %s | FileCheck %s

// CHECK-LABEL:   func.func @test_do_while() -> i32
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.undef : i32
// CHECK:           %[[VAL_2:.*]]:2 = scf.while (%[[VAL_3:.*]] = %[[VAL_0]], %[[VAL_4:.*]] = %[[VAL_1]]) : (i32, i32) -> (i32, i32) {
// CHECK:             %[[VAL_5:.*]]:3 = scf.execute_region -> (i1, i32, i32) {
// CHECK:               %[[VAL_6:.*]] = "get"() : () -> i32
// CHECK:               cf.switch %[[VAL_6]] : i32, [
// CHECK:                 default: ^bb3(%[[VAL_3]] : i32),
// CHECK:                 0: ^bb1,
// CHECK:                 1: ^bb2
// CHECK:               ]
// CHECK:             ^bb1:
// CHECK:               %[[VAL_7:.*]] = "caseA"() : () -> i32
// CHECK:               cf.br ^bb2
// CHECK:             ^bb2:
// CHECK:               %[[VAL_8:.*]] = "caseB"() : () -> i32
// CHECK:               cf.br ^bb3(%[[VAL_8]] : i32)
// CHECK:             ^bb3(%[[VAL_9:.*]]: i32):
// CHECK:               %[[VAL_10:.*]] = "cond"() : () -> i1
// CHECK:               %[[VAL_11:.*]] = arith.constant false
// CHECK:               %[[VAL_12:.*]] = arith.constant true
// CHECK:               %[[VAL_13:.*]] = arith.select %[[VAL_10]], %[[VAL_12]], %[[VAL_11]] : i1
// CHECK:               %[[VAL_14:.*]] = arith.select %[[VAL_10]], %[[VAL_9]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_15:.*]] = arith.select %[[VAL_10]], %[[VAL_9]], %[[VAL_9]] : i32
// CHECK:               scf.yield %[[VAL_13]], %[[VAL_14]], %[[VAL_15]] : i1, i32, i32
// CHECK:             }
// CHECK:             scf.condition(%[[VAL_5]]#0) %[[VAL_5]]#1, %[[VAL_5]]#2 : i32, i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i32):
// CHECK:             scf.yield %[[VAL_17]], %[[VAL_18]] : i32, i32
// CHECK:           }
// CHECK:           return %[[VAL_2]]#1 : i32
// CHECK:         }
func.func @test_do_while() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
  %0 = arith.constant 0 : i32
  cf.br ^bb1(%0 : i32)
^bb1(%1: i32):  // 2 preds: ^bb0, ^bb4
  %2 = "get"() : () -> i32
  cf.switch %2 : i32, [
    default: ^bb4(%1 : i32),
    0: ^bb2,
    1: ^bb3
  ]
^bb2:  // pred: ^bb1
  %3 = "caseA"() : () -> i32
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %4 = "caseB"() : () -> i32
  cf.br ^bb4(%4 : i32)
^bb4(%5: i32):  // 2 preds: ^bb1, ^bb3
  %6 = "cond"() : () -> i1
  cf.cond_br %6, ^bb1(%5 : i32), ^bb5
^bb5:  // pred: ^bb4
  return %5 : i32
}
