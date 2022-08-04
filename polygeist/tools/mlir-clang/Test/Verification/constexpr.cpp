// Copyright (C) Codeplay Software Limited

// RUN: cgeist %s --function=* -S | FileCheck %s

// XFAIL: *

constexpr int num = 10 + 4;

int sum(int*);

int foo() {
    int sz[num];
    for(int i=0; i<num; i++)
        sz[i] = i;
    return sum(sz);
}

// CHECK:   memref.global "private" @_ZL3num : memref<1xi32> = dense<14>
// CHECK:   func @_Z3foov() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = memref.alloca() : memref<14xi32>
// CHECK-NEXT:     %1 = memref.get_global @_ZL3num : memref<1xi32>
// CHECK-NEXT:     %2 = scf.while (%arg0 = %c0_i32) : (i32) -> i32 {
// CHECK-NEXT:       %5 = affine.load %1[0] : memref<1xi32>
// CHECK-NEXT:       %6 = arith.cmpi slt, %arg0, %5 : i32
// CHECK-NEXT:       scf.condition(%6) %arg0 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg0: i32):  // no predecessors
// CHECK-NEXT:       %5 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:       memref.store %arg0, %0[%5] : memref<14xi32>
// CHECK-NEXT:       %6 = arith.addi %arg0, %c1_i32 : i32
// CHECK-NEXT:       scf.yield %6 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = memref.cast %0 : memref<14xi32> to memref<?xi32>
// CHECK-NEXT:     %4 = call @_Z3sumPi(%3) : (memref<?xi32>) -> i32
// CHECK-NEXT:     return %4 : i32
// CHECK-NEXT:   }

