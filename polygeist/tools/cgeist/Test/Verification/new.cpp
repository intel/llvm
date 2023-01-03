// RUN: cgeist -O0 -w %s %stdinclude --function=* -S | FileCheck %s

#include <stdio.h>

struct A {
  float x, y;
};

void f(A *a) { printf("a.x = %f, a.y = %f\n", a->x, a->y); }

int main(int argc, char const *argv[]) {
  // CHECK-DAG:    %c8_i64 = arith.constant 8 : i64
  // CHECK-DAG:    %cst = arith.constant 2.000000e+00 : f32
  // CHECK-DAG:    %cst_0 = arith.constant 1.000000e+00 : f32
  // CHECK-DAG:    %c0_i32 = arith.constant 0 : i32
  // CHECK:        %0 = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr<i8>
  // CHECK-NEXT:   %1 = llvm.bitcast %0 : !llvm.ptr<i8> to !llvm.ptr<struct<(f32, f32)>>
  // CHECK-NEXT:   %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
  // CHECK-NEXT:   llvm.store %cst_0, %2 : !llvm.ptr<f32>
  // CHECK-NEXT:   %3 = llvm.getelementptr %1[0, 1] : (!llvm.ptr<struct<(f32, f32)>>) -> !llvm.ptr<f32>
  // CHECK-NEXT:   llvm.store %cst, %3 : !llvm.ptr<f32>
  // CHECK-NEXT:   call @_Z1fP1A(%1) : (!llvm.ptr<struct<(f32, f32)>>) -> ()
  // CHECK-NEXT:   return %c0_i32 : i32

  auto *a = new A{1.0f, 2.0f};
  f(a);
  return 0;
}
