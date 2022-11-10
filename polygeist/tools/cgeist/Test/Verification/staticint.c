// RUN: cgeist %s --function=* -S | FileCheck %s

int adder(int x) {
    static int cur = 0;
    cur += x;
    return cur;
}

// CHECK-DAG:   memref.global "private" @"adder@static@cur@init" : memref<i1> = dense<true>
// CHECK-DAG:   memref.global "private" @"adder@static@cur" : memref<i32> = dense<0> {alignment = 4 : i64}

// CHECK-LABEL:   func.func @adder(
// CHECK-SAME:                     %[[VAL_0:.*]]: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:        %[[VAL_1:.*]] = arith.constant false
// CHECK-DAG:        %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:       %[[VAL_3:.*]] = memref.get_global @"adder@static@cur" : memref<i32>
// CHECK-NEXT:       %[[VAL_4:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:       %[[VAL_5:.*]] = memref.reshape %[[VAL_3]](%[[VAL_4]]) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:       %[[VAL_6:.*]] = memref.get_global @"adder@static@cur@init" : memref<i1>
// CHECK-NEXT:       %[[VAL_7:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:       %[[VAL_8:.*]] = memref.reshape %[[VAL_6]](%[[VAL_7]]) : (memref<i1>, memref<1xindex>) -> memref<1xi1>
// CHECK-NEXT:       %[[VAL_9:.*]] = affine.load %[[VAL_8]][0] : memref<1xi1>
// CHECK-NEXT:       scf.if %[[VAL_9]] {
// CHECK-NEXT:         affine.store %[[VAL_1]], %[[VAL_8]][0] : memref<1xi1>
// CHECK-NEXT:         affine.store %[[VAL_2]], %[[VAL_5]][0] : memref<1xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[VAL_10:.*]] = affine.load %[[VAL_5]][0] : memref<1xi32>
// CHECK-NEXT:       %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_0]] : i32
// CHECK-NEXT:       affine.store %[[VAL_11]], %[[VAL_5]][0] : memref<1xi32>
// CHECK-NEXT:       return %[[VAL_11]] : i32
// CHECK-NEXT:     }
