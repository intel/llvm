// RUN: cgeist  %s -O3 --function=* -S | FileCheck %s
// COM: Simply check we can lower
// RUN: cgeist  %s -O3 --function=* -S -emit-llvm

void foo(int);

// CHECK-LABEL:   func.func @_Z4loopiii(
// CHECK-SAME:                          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32)
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_6:.*]] = arith.subi %[[VAL_3]], %[[VAL_4]] : index
// CHECK:           %[[VAL_7:.*]] = arith.ceildivui %[[VAL_6]], %[[VAL_5]] : index
// CHECK:           affine.for %[[VAL_8:.*]] = 0 to %[[VAL_7]] {
// CHECK:             %[[VAL_9:.*]] = arith.muli %[[VAL_8]], %[[VAL_5]] : index
// CHECK:             %[[VAL_10:.*]] = arith.divui %[[VAL_9]], %[[VAL_5]] : index
// CHECK:             %[[VAL_11:.*]] = arith.muli %[[VAL_10]], %[[VAL_5]] : index
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_4]], %[[VAL_11]] : index
// CHECK:             %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : index to i32
// CHECK:             func.call @_Z3fooi(%[[VAL_13]]) : (i32) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
void loop(int lb, int ub, int step) {
  for (int x = lb; x < ub; x += step) foo(x);
}
