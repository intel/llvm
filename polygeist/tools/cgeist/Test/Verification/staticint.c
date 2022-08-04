// RUN: cgeist %s --function=* -S | FileCheck %s

int adder(int x) {
    static int cur = 0;
    cur += x;
    return cur;
}

// CHECK:   memref.global "private" @"adder@static@cur@init" : memref<1xi1> = dense<true>
// CHECK:   memref.global "private" @"adder@static@cur" : memref<1xi32> = uninitialized
// CHECK:   func @adder(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %false = arith.constant false
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %0 = memref.get_global @"adder@static@cur" : memref<1xi32>
// CHECK-DAG:     %1 = memref.get_global @"adder@static@cur@init" : memref<1xi1>
// CHECK-NEXT:     %2 = affine.load %1[0] : memref<1xi1>
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       affine.store %false, %1[0] : memref<1xi1>
// CHECK-NEXT:       affine.store %c0_i32, %0[0] : memref<1xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = affine.load %0[0] : memref<1xi32>
// CHECK-NEXT:     %4 = arith.addi %3, %arg0 : i32
// CHECK-NEXT:     affine.store %4, %0[0] : memref<1xi32>
// CHECK-NEXT:     return %4 : i32
// CHECK-NEXT:   }

