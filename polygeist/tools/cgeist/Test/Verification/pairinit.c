// RUN: cgeist %s --function=func -S | FileCheck %s

struct pair {
    int x, y;
};

struct pair func() {
    struct pair tmp = {2, 3};
    return tmp;
}

// CHECK:   func @func(%arg0: memref<?x2xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     affine.store %c2_i32, %arg0[0, 0] : memref<?x2xi32>
// CHECK-NEXT:     affine.store %c3_i32, %arg0[0, 1] : memref<?x2xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
