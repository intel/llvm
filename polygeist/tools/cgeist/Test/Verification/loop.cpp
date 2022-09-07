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

// CHECK:   func @_Z4div_Pi(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.array<25 x struct<(i32, f64)>> : (i64) -> !llvm.ptr<array<25 x struct<(i32, f64)>>>
// CHECK-NEXT:     %1 = memref.get_global @MAX_DIMS : memref<1xi32>
// CHECK-NEXT:     %2 = affine.load %1[0] : memref<1xi32>
// CHECK-NEXT:     %3 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<array<25 x struct<(i32, f64)>>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %4 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:     scf.for %arg1 = %c0 to %4 step %c1 {
// CHECK-NEXT:       %5 = arith.index_cast %arg1 : index to i64
// CHECK-NEXT:       %6 = llvm.getelementptr %3[%5] : (!llvm.ptr<struct<(i32, f64)>>, i64) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:       %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:       %8 = memref.load %arg0[%arg1] : memref<?xi32>
// CHECK-NEXT:       llvm.store %8, %7 : !llvm.ptr<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
