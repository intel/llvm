// RUN: cgeist  -O0 -w %s %stdinclude --function=* -S | FileCheck %s

#include <stdio.h>

struct A {
  float x, y;
};

void f(A *a) { printf("a.x = %f, a.y = %f\n", a->x, a->y); }

int main(int argc, char const *argv[]) {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<1x!llvm.struct<(f32, f32)>>
// CHECK:           %[[VAL_6:.*]] = "polygeist.memref2pointer"(%[[ALLOC]]) : (memref<1x!llvm.struct<(f32, f32)>>) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_6]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
// CHECK-NEXT:      llvm.store %[[VAL_5]], %[[VAL_7]] : f32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_6]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
// CHECK-NEXT:      llvm.store %[[VAL_4]], %[[VAL_8]] : f32, !llvm.ptr
// CHECK-NEXT:      call @_Z1fP1A(%[[VAL_6]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return %[[VAL_3]] : i32
// CHECK-NEXT:    }

  auto *a = new A{1.0f, 2.0f};
  f(a);
  return 0;
}
