// RUN: cgeist %s -w -O0 --function=* -S | FileCheck %s

// CHECK-LABEL:   func.func @_Z14lambda_capturebb(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i1)
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(i8, memref<?xi8>)> : (i64) -> !llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(i8, memref<?xi8>)> : (i64) -> !llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_4]][0, 0] : (!llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>) -> !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.extui %[[VAL_0]] : i1 to i8
// CHECK-NEXT:      llvm.store %[[VAL_6]], %[[VAL_5]] : !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.extui %[[VAL_1]] : i1 to i8
// CHECK-NEXT:      %[[VAL_8:.*]] = memref.alloca() : memref<1xi8>
// CHECK-NEXT:      affine.store %[[VAL_7]], %[[VAL_8]][0] : memref<1xi8>
// CHECK-NEXT:      %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_4]][0, 1] : (!llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>) -> !llvm.ptr<memref<?xi8>>
// CHECK-NEXT:      llvm.store %[[VAL_9]], %[[VAL_10]] : !llvm.ptr<memref<?xi8>>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>
// CHECK-NEXT:      llvm.store %[[VAL_11]], %[[VAL_3]] : !llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>
// CHECK-NEXT:      call @_ZZ14lambda_capturebbENK3$_0clEv(%[[VAL_3]]) : (!llvm.ptr<!llvm.struct<(i8, memref<?xi8>)>>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

void lambda_capture(bool x, bool y) {
  const auto f = [x, &y]() { y = y ^ x; };
  f();
}
