// RUN: cgeist %s --function=* -S -O0 -w | FileCheck %s

// CHECK-LABEL:   func.func @foldTrunc() -> f32
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 1.000000e+01 : f32
// CHECK-NEXT:      return %[[VAL_0]] : f32
// CHECK-NEXT:    }
float foldTrunc() {
  double d = 10;
  return d;
}
