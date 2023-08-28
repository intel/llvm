// RUN: cgeist %s --function=* -S | FileCheck %s

void foo();
void bar();

// CHECK-LABEL:   func.func @_Z9elvisVoidb(
// CHECK-SAME:                             %[[VAL_0:.*]]: i1)
// CHECK-NEXT:      scf.if %[[VAL_0]] {
// CHECK-NEXT:        func.call @_Z3foov() : () -> ()
// CHECK-NEXT:      } else {
// CHECK-NEXT:        func.call @_Z3barv() : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void elvisVoid(bool cond) {
  cond ? foo() : bar();
}

// CHECK-LABEL:   func.func @_Z11elvisRValuebii(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                  %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                  %[[VAL_2:.*]]: i32) -> i32
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.select %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : i32
// CHECK-NEXT:      return %[[VAL_3]] : i32
// CHECK-NEXT:    }
int elvisRValue(bool cond, int x, int y) {
  return cond ? x : y;
}

// CHECK-LABEL:   func.func @_Z11elvisLValuebRiS_(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                    %[[VAL_1:.*]]: memref<?xi32>,
// CHECK-SAME:                                    %[[VAL_2:.*]]: memref<?xi32>)
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.select %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : memref<?xi32>
// CHECK-NEXT:      affine.store %[[VAL_3]], %[[VAL_4]][0] : memref<?xi32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void elvisLValue(bool cond, int &a, int &b) {
  (cond ? a : b) = 0;
}
