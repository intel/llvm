// RUN: cgeist %s -w -O0 --function=* -S | FileCheck %s

// CHECK-LABEL:   func.func @_Z14lambda_capturebb(
// CHECK-SAME:                                     %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                     %[[VAL_1:.*]]: i1)
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(i8, memref<?xi8>)> : (i64) -> !llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_3]][0, 0] : (!llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>) -> !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_5:.*]] = arith.extui %[[VAL_0]] : i1 to i8
// CHECK-NEXT:      llvm.store %[[VAL_5]], %[[VAL_4]] : !llvm.ptr<i8>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void lambda_capture(bool x, bool y) {
  const auto f = [x, &y]() { y = !x; };
}
