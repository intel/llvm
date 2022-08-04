// RUN: cgeist %s %stdinclude --function=* -S | FileCheck %s

#include <stdio.h>

struct A {
  float x, y;
};

void f(A *a) { printf("a.x = %f, a.y = %f\n", a->x, a->y); }

int main(int argc, char const *argv[]) {
  // CHECK-DAG: %[[two:.*]] = arith.constant 2.000000e+00 : f32
  // CHECK-DAG: %[[one:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<1x2xf32>
  // CHECK: affine.store %[[one]], %[[alloc]][0, 0] : memref<1x2xf32>
  // CHECK: affine.store %[[two]], %[[alloc]][0, 1] : memref<1x2xf32>
  auto *a = new A{1.0f, 2.0f};
  f(a);
  return 0;
}
