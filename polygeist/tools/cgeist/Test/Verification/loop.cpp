// RUN: cgeist  %s -O2 --function=* -S | FileCheck %s

int MAX_DIMS;
static short S;

struct A {
  int x;
  double y;
};

void div_(int* sizes, short k) {
  A data[25];
  S = k;
  for (int i=0; i < MAX_DIMS; ++i) {
    data[i].x = sizes[i] + S;
  }
}

// CHECK-LABEL:   memref.global @MAX_DIMS : memref<i32> = dense<0> {alignment = 4 : i64}
// CHECK-NEXT:    memref.global "private" @_ZL1S : memref<i16> = dense<0> {alignment = 2 : i64}

// CHECK-LABEL:   func.func @_Z4div_Pis(
// CHECK-SAME:                          %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: i16) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x !llvm.array<25 x struct<(i32, f64)>> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = memref.get_global @_ZL1S : memref<i16>
// CHECK-NEXT:      %[[VAL_7:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      %[[VAL_8:.*]] = memref.reshape %[[VAL_6]](%[[VAL_7]]) : (memref<i16>, memref<1xindex>) -> memref<1xi16>
// CHECK-NEXT:      affine.store %[[VAL_1]], %[[VAL_8]][0] : memref<1xi16>
// CHECK-NEXT:      %[[VAL_9:.*]] = memref.get_global @MAX_DIMS : memref<i32>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_5]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x struct<(i32, f64)>>
// CHECK-NEXT:      %[[VAL_11:.*]] = scf.while (%[[VAL_12:.*]] = %[[VAL_3]]) : (i32) -> i32 {
// CHECK-NEXT:        %[[VAL_13:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:        %[[VAL_14:.*]] = memref.reshape %[[VAL_9]](%[[VAL_13]]) : (memref<i32>, memref<1xindex>) -> memref<1xi32>
// CHECK-NEXT:        %[[VAL_15:.*]] = affine.load %[[VAL_14]][0] : memref<1xi32>
// CHECK-NEXT:        %[[VAL_16:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_15]] : i32
// CHECK-NEXT:        scf.condition(%[[VAL_16]]) %[[VAL_12]] : i32
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%[[VAL_17:.*]]: i32):
// CHECK-NEXT:        %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : i32 to index
// CHECK-NEXT:        %[[VAL_19:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_18]]] : memref<?xi32>
// CHECK-NEXT:        %[[VAL_20:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:        %[[VAL_21:.*]] = memref.reshape %[[VAL_6]](%[[VAL_20]]) : (memref<i16>, memref<1xindex>) -> memref<1xi16>
// CHECK-NEXT:        %[[VAL_22:.*]] = affine.load %[[VAL_21]][0] : memref<1xi16>
// CHECK-NEXT:        %[[VAL_23:.*]] = arith.extsi %[[VAL_22]] : i16 to i32
// CHECK-NEXT:        %[[VAL_24:.*]] = arith.addi %[[VAL_19]], %[[VAL_23]] : i32
// CHECK-NEXT:        %[[VAL_25:.*]] = arith.index_cast %[[VAL_18]] : index to i64
// CHECK-NEXT:        %[[VAL_26:.*]] = llvm.getelementptr %[[VAL_10]]{{\[}}%[[VAL_25]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, f64)>
// CHECK-NEXT:        %[[VAL_27:.*]] = llvm.getelementptr inbounds %[[VAL_26]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
// CHECK-NEXT:        llvm.store %[[VAL_24]], %[[VAL_27]] : i32, !llvm.ptr
// CHECK-NEXT:        %[[VAL_28:.*]] = arith.addi %[[VAL_17]], %[[VAL_2]] : i32
// CHECK-NEXT:        scf.yield %[[VAL_28]] : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
