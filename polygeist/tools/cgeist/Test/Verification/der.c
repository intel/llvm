// RUN: cgeist %s --function=kernel_deriche -S | FileCheck %s

float kernel_deriche() {
    float a2, a6;
    a2 = a6 = 2.0;//EXP_FUN(-alpha);
    return a2;
}

// CHECK:  func @kernel_deriche() -> f32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:    return %cst : f32
// CHECK-NEXT:  }
