// RUN: cgeist %s --function=* -S -o - | FileCheck %s

// COM: Checking evaluation order of binary operations.
// COM: Each operation must be further tested in separate files.

// CHECK-LABEL: func.func @f0(%arg0: memref<?xi32>, %arg1: memref<?xi32>)
// CHECK-DAG:     %[[LHS:.*]] = affine.load %arg0[0] : memref<?xi32>
// CHECK-DAG:     %[[RHS:.*]] = affine.load %arg1[1] : memref<?xi32>
// CHECK-NEXT:    %[[RET:.*]] = arith.muli %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:    return %[[RET]] : i32
// CHECK-NEXT:  }

int f0(int *x, int *y) {
  int i = 0;
  return x[i++] * y[i];
}

// CHECK-LABEL: func.func @f1(%arg0: memref<?xi32>, %arg1: memref<?xi32>)
// CHECK-DAG:     %[[LHS:.*]] = affine.load %arg0[1] : memref<?xi32>
// CHECK-DAG:     %[[RHS:.*]] = affine.load %arg1[1] : memref<?xi32>
// CHECK-NEXT:    %[[RET:.*]] = arith.shli %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:    return %[[RET]] : i32
// CHECK-NEXT:  }

int f1(int *x, int *y) {
  int i = 0;
  return x[++i] << y[i];
}

// CHECK-LABEL: func.func @f2(%arg0: memref<?xi32>, %arg1: memref<?xi32>)
// CHECK-DAG:     %[[LHS:.*]] = affine.load %arg0[0] : memref<?xi32>
// CHECK-DAG:     %[[RHS:.*]] = affine.load %arg1[1] : memref<?xi32>
// CHECK-NEXT:    %[[RET:.*]] = arith.remsi %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:    return %[[RET]] : i32
// CHECK-NEXT:  }

int f2(int *x, int *y) {
  int i = 0;
  return x[i] % y[++i];
}

// CHECK-LABEL: func.func @f3(%arg0: memref<?xi32>, %arg1: memref<?xi32>)
// CHECK-DAG:     %[[LHS:.*]] = affine.load %arg0[0] : memref<?xi32>
// CHECK-DAG:     %[[RHS:.*]] = affine.load %arg1[0] : memref<?xi32>
// CHECK-NEXT:    %[[RET:.*]] = arith.xori %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:    return %[[RET]] : i32
// CHECK-NEXT:  }

int f3(int *x, int *y) {
  int i = 0;
  return x[i] ^ y[i++];
}
