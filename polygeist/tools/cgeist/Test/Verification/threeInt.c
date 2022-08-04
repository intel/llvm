// RUN: cgeist %s --function=struct_pass_all_same -S | FileCheck %s

typedef struct {
  int a, b, c;
} threeInt;

int struct_pass_all_same(threeInt* a) {
  return a->b;
}

// CHECK:  func @struct_pass_all_same(%arg0: memref<?x3xi32>) -> i32
// CHECK-NEXT:    %0 = affine.load %arg0[0, 1] : memref<?x3xi32>
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
