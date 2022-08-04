// Copyright (C) Codeplay Software Limited

// RUN: cgeist %s --function=create_matrix -S | FileCheck %s

// XFAIL: *

void create_matrix(float *m, int size) {
  float coe[2 * size - 1];
  coe[size] = 0;
  m[size] = coe[size];
}

// CHECK:   func @create_matrix(%arg0: memref<?xf32>, %arg1: i32)
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.muli %0, %c2 : index
// CHECK-NEXT:     %2 = arith.subi %1, %c1 : index
// CHECK-NEXT:     %3 = memref.alloca(%2) : memref<?xf32>
// CHECK-NEXT:     affine.store %cst, %3[symbol(%0)] : memref<?xf32>
// CHECK-NEXT:     %[[i5:.+]] = affine.load %3[symbol(%0)] : memref<?xf32>
// CHECK-NEXT:     affine.store %[[i5]], %arg0[symbol(%0)] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
