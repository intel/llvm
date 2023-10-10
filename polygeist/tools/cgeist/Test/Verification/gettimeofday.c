// RUN: cgeist -O0 -w %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// CHECK:         func.func @alloc() -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = call @gettimeofday(%[[VAL_2]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.sitofp %[[VAL_6]] : i64 to f64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.sitofp %[[VAL_9]] : i64 to f64
// CHECK-NEXT:      %[[VAL_11:.*]] = arith.mulf %[[VAL_10]], %[[VAL_0]] : f64
// CHECK-NEXT:      %[[VAL_12:.*]] = arith.addf %[[VAL_7]], %[[VAL_11]] : f64
// CHECK-NEXT:      return %[[VAL_12]] : f64
// CHECK-NEXT:    }
