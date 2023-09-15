// RUN: cgeist %s --function=* -S -o - | FileCheck %s

// CHECK-LABEL: func.func @f0(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    affine.store %c0_i32, %arg0[0] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

void f0(int *x) {
  int i = 0;
  x[i++] = i;
}

// CHECK-LABEL: func.func @f1(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    affine.store %c0_i32, %arg0[1] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

void f1(int *x) {
  int i = 0;
  x[++i] = i;
}

// CHECK-LABEL: func.func @f2(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    affine.store %c1_i32, %arg0[1] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

void f2(int *x) {
  int i = 0;
  x[i] = ++i;
}

// CHECK-LABEL: func.func @f3(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    affine.store %c0_i32, %arg0[1] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

void f3(int *x) {
  int i = 0;
  x[i] = i++;
}

// CHECK-LABEL: func.func @f4(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    affine.store %c1_i32, %arg0[0] : memref<?xi32>
// CHECK-NEXT:    affine.store %c1_i32, %arg1[0] : memref<?xi32>
// CHECK-NEXT:    return %c1_i32 : i32
// CHECK-NEXT:  }

int f4(int *x, int *y) {
  return *y = *x = 1;
}
