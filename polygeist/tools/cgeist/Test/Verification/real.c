// RUN: cgeist %s --function=* -S | FileCheck %s

#include <complex.h>

// CHECK-LABEL:   func.func @f0(
// CHECK-SAME:                  %[[VAL_0:.*]]: i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

int f0(int a) {
  return __real__(a);
}

// CHECK-LABEL:   func.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: f32) -> f32
// CHECK:           return %[[VAL_0]] : f32
// CHECK:         }

float f1(float a) {
  return __real__(a);
}
