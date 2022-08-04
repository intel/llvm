// RUN: cgeist %s %stdinclude --function=alloc -S | FileCheck %s

// XFAIL: *

#include <time.h>
#include <sys/time.h>
double alloc() {
  struct timeval Tp;
  gettimeofday(&Tp, NULL);
  return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

// CHECK:   func @alloc() -> f64
// CHECK-NEXT:     %cst = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x2xi64>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%0) : (memref<1x2xi64>) -> !llvm.ptr<struct<(i64, i64)>>
// CHECK-NEXT:     %2 = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:     %3 = llvm.bitcast %2 : !llvm.ptr<i8> to !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:     %4 = llvm.call @gettimeofday(%1, %3) : (!llvm.ptr<struct<(i64, i64)>>, !llvm.ptr<struct<(i32, i32)>>) -> i32
// CHECK-NEXT:     %5 = affine.load %0[0, 0] : memref<1x2xi64>
// CHECK-NEXT:     %6 = arith.sitofp %5 : i64 to f64
// CHECK-NEXT:     %7 = affine.load %0[0, 1] : memref<1x2xi64>
// CHECK-NEXT:     %8 = arith.sitofp %7 : i64 to f64
// CHECK-NEXT:     %9 = arith.mulf %8, %cst : f64
// CHECK-NEXT:     %10 = arith.addf %6, %9 : f64
// CHECK-NEXT:     return %10 : f64
// CHECK-NEXT:   }
