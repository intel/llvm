// RUN: cgeist %s --function=gcd -S | FileCheck %s

int gcd(int m, int n) {
  while (n > 0) {
    int r = m % n;
    m = n;
    n = r;
  }
  return m;
}

// CHECK:   func @gcd(%arg0: i32, %arg1: i32) -> i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0:2 = scf.while (%arg2 = %arg1, %arg3 = %arg0) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:       %1 = arith.cmpi sgt, %arg2, %c0_i32 : i32
// CHECK-NEXT:       scf.condition(%1) %arg3, %arg2 : i32, i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg2: i32, %arg3: i32):
// CHECK-NEXT:       %1 = arith.remsi %arg2, %arg3 : i32
// CHECK-NEXT:       scf.yield %1, %arg3 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#0 : i32
// CHECK-NEXT:   }
