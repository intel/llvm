// RUN: cgeist  %s -O0 --function=* -S | FileCheck %s

extern "C" {

double kernel_deriche(int x, float y) {
    ([&y,x]() {
        y *= x;
    })();
    return y;
}

}

// CHECK-LABEL:   func.func @kernel_deriche(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32,
// CHECK-SAME:                              %[[VAL_1:.*]]: f32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(memref<?xf32>, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(memref<?xf32>, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.mlir.undef : f32
// CHECK-NEXT:      affine.store %[[VAL_1]], %[[VAL_5]][0] : memref<1xf32>
// CHECK-NEXT:      %[[VAL_7:.*]] = memref.cast %[[VAL_5]] : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xf32>, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_7]], %[[VAL_8]] : memref<?xf32>, !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xf32>, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_9]] : i32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> !llvm.struct<(memref<?xf32>, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_10]], %[[VAL_3]] : !llvm.struct<(memref<?xf32>, i32)>, !llvm.ptr
// CHECK-NEXT:      call @_ZZ14kernel_dericheENK3$_0clEv(%[[VAL_3]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_11:.*]] = affine.load %[[VAL_5]][0] : memref<1xf32>
// CHECK-NEXT:      %[[VAL_12:.*]] = arith.extf %[[VAL_11]] : f32 to f64
// CHECK-NEXT:      return %[[VAL_12]] : f64
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func private @_ZZ14kernel_dericheENK3$_0clEv(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xf32>, i32)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i32
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.sitofp %[[VAL_2]] : i32 to f32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xf32>, i32)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> memref<?xf32>
// CHECK-NEXT:      %[[VAL_6:.*]] = affine.load %[[VAL_5]][0] : memref<?xf32>
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.mulf %[[VAL_6]], %[[VAL_3]] : f32
// CHECK-NEXT:      affine.store %[[VAL_7]], %[[VAL_5]][0] : memref<?xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
