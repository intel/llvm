// RUN: cgeist --use-opaque-pointers -O0 -w %s %stdinclude --function=* -S | FileCheck %s

#include <stdio.h>

struct A {
  float x, y;
};

void f(A *a) { printf("a.x = %f, a.y = %f\n", a->x, a->y); }

int main(int argc, char const *argv[]) {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = "polygeist.typeSize"() {source = !polygeist.struct<(f32, f32)>} : () -> index
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : index to i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.call @malloc(%[[VAL_6]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !polygeist.struct<(f32, f32)>
// CHECK-NEXT:      llvm.store %[[VAL_4]], %[[VAL_8]] : f32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !polygeist.struct<(f32, f32)>
// CHECK-NEXT:      llvm.store %[[VAL_3]], %[[VAL_9]] : f32, !llvm.ptr
// CHECK-NEXT:      call @_Z1fP1A(%[[VAL_7]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return %[[VAL_2]] : i32
// CHECK-NEXT:    }

  auto *a = new A{1.0f, 2.0f};
  f(a);
  return 0;
}
