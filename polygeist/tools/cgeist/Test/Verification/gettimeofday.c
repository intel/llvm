// RUN: cgeist -O0 -w %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// CHECK:   func @alloc() -> f64
// CHECK-NEXT:     %cst = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr<struct<(i64, i64)>>
// CHECK-NEXT:     %1 = llvm.mlir.null : !llvm.ptr<{{.*}}>
// CHECK:          %{{.*}} = llvm.call @gettimeofday(%0, %{{.*}}) : (!llvm.ptr<struct<(i64, i64)>>, {{.*}}) -> i32
// CHECK-NEXT:     [[T3:%.*]] = llvm.getelementptr inbounds %0[0, 0] : (!llvm.ptr<struct<(i64, i64)>>) -> !llvm.ptr<i64>
// CHECK-NEXT:     [[T4:%.*]] = llvm.load [[T3]] : !llvm.ptr<i64>
// CHECK-NEXT:     [[T5:%.*]] = arith.sitofp [[T4]] : i64 to f64
// CHECK-NEXT:     [[T6:%.*]] = llvm.getelementptr inbounds %0[0, 1] : (!llvm.ptr<struct<(i64, i64)>>) -> !llvm.ptr<i64>
// CHECK-NEXT:     [[T7:%.*]] = llvm.load [[T6]] : !llvm.ptr<i64>
// CHECK-NEXT:     [[T8:%.*]] = arith.sitofp [[T7]] : i64 to f64
// CHECK-NEXT:     [[T9:%.*]] = arith.mulf [[T8]], %cst : f64
// CHECK-NEXT:     [[T10:%.*]] = arith.addf [[T5]], [[T9]] : f64
// CHECK-NEXT:     return [[T10]] : f64
// CHECK-NEXT:   }
