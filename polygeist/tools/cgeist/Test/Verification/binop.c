// RUN: cgeist %s --function=* -S -o - | FileCheck %s

// COM: Checking evaluation order of binary operations.
// COM: Each operation must be further tested in separate files.

// CHECK-LABEL: func.func @f0(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<?xi32>
// CHECK-NEXT:    %1 = affine.load %arg1[1] : memref<?xi32>
// CHECK-NEXT:    %2 = arith.muli %0, %1 : i32
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }

int f0(int *x, int *y) {
  int i = 0;
  return x[i++] * y[i];
}

// CHECK-LABEL: func.func @f1(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %0 = affine.load %arg0[1] : memref<?xi32>
// CHECK-NEXT:    %1 = affine.load %arg1[1] : memref<?xi32>
// CHECK-NEXT:    %2 = arith.shli %0, %1 : i32
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }

int f1(int *x, int *y) {
  int i = 0;
  return x[++i] << y[i];
}

// CHECK-LABEL: func.func @f2(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<?xi32>
// CHECK-NEXT:    %1 = affine.load %arg1[1] : memref<?xi32>
// CHECK-NEXT:    %2 = arith.remsi %0, %1 : i32
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }

int f2(int *x, int *y) {
  int i = 0;
  return x[i] % y[++i];
}

// CHECK-LABEL: func.func @f3(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<?xi32>
// CHECK-NEXT:    %1 = affine.load %arg1[0] : memref<?xi32>
// CHECK-NEXT:    %2 = arith.xori %0, %1 : i32
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }

int f3(int *x, int *y) {
  int i = 0;
  return x[i] ^ y[i++];
}
