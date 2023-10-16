// RUN: cgeist %s --function=caller %stdinclude -S | FileCheck %s

#include <stdlib.h>

void sum(double *result);

void caller(int size) {
    double* array = (double*)malloc(sizeof(double) * size);
    sum(array);
    free(array);
}

// CHECK-LABEL:   func.func @caller(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32)
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 8 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i32 to i64
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.muli %[[VAL_2]], %[[VAL_1]] : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = call @malloc(%[[VAL_3]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = "polygeist.pointer2memref"(%[[VAL_4]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:      call @sum(%[[VAL_5]]) : (memref<?xf64>) -> ()
// CHECK-NEXT:      memref.dealloc %[[VAL_5]] : memref<?xf64>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
