// RUN: cgeist %s --function=* -S | FileCheck %s

#include <complex.h>

// CHECK-LABEL:   func.func @f0(
// CHECK-SAME:                  %[[VAL_0:.*]]: i32) -> i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_1]] : i32
// CHECK:         }

int f0(int a) {
  return __imag__(a);
}

// CHECK-LABEL:   func.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: f32) -> f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           return %[[VAL_1]] : f32
// CHECK:         }

float f1(float a) {
  return __imag__(a);
}
