// RUN: cgeist %s --function=* -S | FileCheck %s

int adder(int x) {
    static int cur = 0;
    cur += x;
    return cur;
}

// CHECK-DAG:   memref.global "private" @"adder@static@cur@init" : memref<1xi1> = dense<true>
// CHECK-DAG:   memref.global "private" @"adder@static@cur" : memref<i32> = uninitialized

// CHECK-LABEL:   func.func @adder(
// CHECK-SAME:                     %[[VAL_0:.*]]: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:        %[[VAL_1:.*]] = arith.constant false
// CHECK-DAG:        %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:       %[[VAL_3:.*]] = memref.get_global @"adder@static@cur" : memref<i32>
// CHECK-NEXT:       %[[VAL_4:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:       %[[VAL_5:.*]] = memref.reshape %[[VAL_3]](%[[VAL_4]]) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:       %[[VAL_6:.*]] = memref.get_global @"adder@static@cur@init" : memref<1xi1>
// CHECK-NEXT:       %[[VAL_7:.*]] = affine.load %[[VAL_6]][0] : memref<1xi1>
// CHECK-NEXT:       scf.if %[[VAL_7]] {
// CHECK-NEXT:         affine.store %[[VAL_1]], %[[VAL_6]][0] : memref<1xi1>
// CHECK-NEXT:         affine.store %[[VAL_2]], %[[VAL_5]][0] : memref<1xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[VAL_8:.*]] = affine.load %[[VAL_5]][0] : memref<1xi32>
// CHECK-NEXT:       %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_0]] : i32
// CHECK-NEXT:       affine.store %[[VAL_9]], %[[VAL_5]][0] : memref<1xi32>
// CHECK-NEXT:       return %[[VAL_9]] : i32
// CHECK-NEXT:     }
