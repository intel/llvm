// RUN: cgeist %s %stdinclude --function=alloc -S | FileCheck %s

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// clang-format off
// CHECK:   func @alloc() -> f64
// CHECK-NEXT:     %cst = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:     %alloca = memref.alloca() : memref<1x2xi64>
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x2xi64>) -> !llvm.ptr<struct<(i64, i64)>>
// CHECK-NEXT:     %1 = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:          %{{.*}} = llvm.call @gettimeofday(%0, %{{.*}}) : (!llvm.ptr<struct<(i64, i64)>>, {{.*}}) -> i32
// CHECK-NEXT:     [[T5:%.*]] = affine.load %alloca[0, 0] : memref<1x2xi64>
// CHECK-NEXT:     [[T6:%.*]]  = arith.sitofp [[T5]] : i64 to f64
// CHECK-NEXT:     [[T7:%.*]]  = affine.load %alloca[0, 1] : memref<1x2xi64>
// CHECK-NEXT:     [[T8:%.*]]  = arith.sitofp [[T7]] : i64 to f64
// CHECK-NEXT:     [[T9:%.*]]  = arith.mulf [[T8]], %cst : f64
// CHECK-NEXT:     [[T10:%.*]]  = arith.addf [[T6]], [[T9]] : f64
// CHECK-NEXT:     return [[T10]] : f64
// CHECK-NEXT:   }
// clang-format on
