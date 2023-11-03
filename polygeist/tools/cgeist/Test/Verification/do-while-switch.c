// RUN: cgeist -o - -S --function=test_do_while %s | FileCheck %s

enum Kind { A, B };

enum Kind get();
int cond();
int caseA();
int caseB();

// CHECK-LABEL:   func.func @test_do_while() -> i32
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.undef : i32
// CHECK:           %[[VAL_2:.*]]:2 = scf.while (%[[VAL_3:.*]] = %[[VAL_1]]) : (i32) -> (i32, i32) {
// CHECK:             %[[VAL_4:.*]]:3 = scf.execute_region -> (i1, i32, i32) {
// CHECK:               %[[VAL_5:.*]] = func.call @get() : () -> i32
// CHECK:               cf.switch %[[VAL_5]] : i32, [
// CHECK:                 default: ^bb3(%[[VAL_3]] : i32),
// CHECK:                 0: ^bb2,
// CHECK:                 1: ^bb1
// CHECK:               ]
// CHECK:             ^bb1:
// CHECK:               %[[VAL_6:.*]] = func.call @caseB() : () -> i32
// CHECK:               cf.br ^bb3(%[[VAL_6]] : i32)
// CHECK:             ^bb2:
// CHECK:               %[[VAL_7:.*]] = func.call @caseA() : () -> i32
// CHECK:               cf.br ^bb3(%[[VAL_7]] : i32)
// CHECK:             ^bb3(%[[VAL_8:.*]]: i32):
// CHECK:               %[[VAL_9:.*]] = func.call @cond() : () -> i32
// CHECK:               %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_0]] : i32
// CHECK:               %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:               scf.yield %[[VAL_10]], %[[VAL_11]], %[[VAL_8]] : i1, i32, i32
// CHECK:             }
// CHECK:             scf.condition(%[[VAL_4]]#0) %[[VAL_4]]#1, %[[VAL_4]]#2 : i32, i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32):
// CHECK:             scf.yield %[[VAL_13]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_2]]#1 : i32
// CHECK:         }
int test_do_while() {
  int res;
  do {
    switch (get()) {
    case A:
      res = caseA();
      break;

    case B:
      res = caseB();
      break;
    }
  } while (cond());
  return res;
}
