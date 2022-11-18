// RUN: cgeist %s -O2 --function=create_matrix -S | FileCheck %s

void create_matrix(float *m, int size) {
  float coe[2 * size + 1];
  coe[size] = 1.0;
  m[size] = coe[size] + coe[0];
}

// CHECK:   func @create_matrix(%arg0: memref<?xf32>, %arg1: i32)
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %0 = arith.muli %arg1, %c2_i32 : i32
// CHECK-NEXT:     %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:     %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:     %alloca = memref.alloca(%2) : memref<?xf32>
// CHECK-NEXT:     %3 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     affine.store %cst, %alloca[symbol(%3)] : memref<?xf32>
// CHECK-NEXT:     %[[i5:.+]] = affine.load %alloca[0] : memref<?xf32>
// CHECK-NEXT:     %[[i6:.+]] = arith.addf %[[i5]], %cst : f32
// CHECK-NEXT:     affine.store %[[i6]], %arg0[symbol(%3)] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
