// RUN: cgeist %s --function=test -S | FileCheck %s

int test() {
    return -3;
}

// CHECK:  func @test() -> i32
// CHECK-NEXT:    %c-3_i32 = arith.constant -3 : i32
// CHECK-NEXT:    return %c-3_i32 : i32
// CHECK-NEXT:  }
