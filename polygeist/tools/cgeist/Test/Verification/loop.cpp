// RUN: cgeist %s --function=* -S | FileCheck %s

int MAX_DIMS;

struct A {
    int x;
    double y;
};

void div_(int* sizes) {
    A data[25];
    for (int i=0; i < MAX_DIMS; ++i) {
            data[i].x = sizes[i];
    }
}

// CHECK-LABEL:   memref.global @MAX_DIMS : memref<i32> = uninitialized {alignment = 4 : i64}

// CHECK-LABEL:   func.func @_Z4div_Pi(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:        %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:        %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:        %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.array<25 x struct<(i32, f64)>> : (i64) -> !llvm.ptr<array<25 x struct<(i32, f64)>>>
// CHECK-NEXT:       %[[VAL_5:.*]] = memref.get_global @MAX_DIMS : memref<i32>
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_4]][0, 0] : (!llvm.ptr<array<25 x struct<(i32, f64)>>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:       %[[VAL_7:.*]] = scf.while (%[[VAL_8:.*]] = %[[VAL_2]]) : (i32) -> i32 {
// CHECK-NEXT:         %[[VAL_9:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:         %[[VAL_10:.*]] = memref.reshape %[[VAL_5]](%[[VAL_9]]) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:         %[[VAL_11:.*]] = affine.load %[[VAL_10]][0] : memref<1xi32>
// CHECK-NEXT:         %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_11]] : i32
// CHECK-NEXT:         scf.condition(%[[VAL_12]]) %[[VAL_8]] : i32
// CHECK-NEXT:       } do {
// CHECK-NEXT:       ^bb0(%[[VAL_13:.*]]: i32):
// CHECK-NEXT:         %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : i32 to index
// CHECK-NEXT:         %[[VAL_15:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_14]]] : memref<?xi32>
// CHECK-NEXT:         %[[VAL_16:.*]] = arith.index_cast %[[VAL_14]] : index to i64
// CHECK-NEXT:         %[[VAL_17:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_16]]] : (!llvm.ptr<struct<(i32, f64)>>, i64) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:         %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_17]][0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:         llvm.store %[[VAL_15]], %[[VAL_18]] : !llvm.ptr<i32>
// CHECK-NEXT:         %[[VAL_19:.*]] = arith.addi %[[VAL_13]], %[[VAL_1]] : i32
// CHECK-NEXT:         scf.yield %[[VAL_19]] : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
