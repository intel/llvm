// RUN: cgeist %s --function=foo -S | FileCheck %s

void foo(int A[10], int a) {
  for (int i = 0; i < 10; ++i)
    A[i] ^= (a ^ (A[i] + 1));
}

// CHECK: func @foo(%arg0: memref<?xi32>, %arg1: i32)
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c10 = arith.constant 10 : index
// CHECK-NEXT:     scf.for %arg2 = %c0 to %c10 step %c1 {
// CHECK-NEXT:       %0 = memref.load %arg0[%arg2] : memref<?xi32>
// CHECK-NEXT:       %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:       %2 = arith.xori %arg1, %1 : i32
// CHECK-NEXT:       %3 = arith.xori %0, %2 : i32
// CHECK-NEXT:       memref.store %3, %arg0[%arg2] : memref<?xi32>
// CHECK-NEXT:     }
