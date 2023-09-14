// RUN: cgeist %s --function=kernel_deriche -S | FileCheck %s

int kernel_deriche(int x) {
    x++;
    x+=3;
    x*=2;
    return x;
}

// CHECK:  func @kernel_deriche(%arg0: i32) -> i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c4_i32 = arith.constant 4 : i32
// CHECK:     %0 = arith.addi %arg0, %c4_i32 : i32
// CHECK:     %1 = arith.muli %0, %c2_i32 : i32
// CHECK:     return %1 : i32
// CHECK-NEXT:   }
