// RUN: cgeist %s -O2 --function=* -S | FileCheck %s

extern "C" {

double kernel_deriche(int x, float y) {
    ([&y,x]() {
        y *= x;
    })();
    return y;
}

}

// CHECK:   func @kernel_deriche(%arg0: i32, %arg1: f32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(memref<?xf32>, i32)> : (i64) -> !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %1 = llvm.alloca %c1_i64 x !llvm.struct<(memref<?xf32>, i32)> : (i64) -> !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     %alloca = memref.alloca() : memref<1xf32>
// CHECK-NEXT:     affine.store %arg1, %alloca[0] : memref<1xf32>
// CHECK-NEXT:     %cast = memref.cast %alloca : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     llvm.store %cast, %2 : !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %3 = llvm.getelementptr %1[0, 1] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %arg0, %3 : !llvm.ptr<i32>
// CHECK-NEXT:     %4 = llvm.load %1 : !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     llvm.store %4, %0 : !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>
// CHECK-NEXT:     call @_ZZ14kernel_dericheENK3$_0clEv(%0) : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> ()
// CHECK-NEXT:     %5 = affine.load %alloca[0] : memref<1xf32>
// CHECK-NEXT:     %6 = arith.extf %5 : f32 to f64
// CHECK-NEXT:     return %6 : f64
// CHECK-NEXT:   }
// CHECK:   func private @_ZZ14kernel_dericheENK3$_0clEv(%arg0: !llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<i32>
// CHECK-NEXT:     %2 = arith.sitofp %1 : i32 to f32
// CHECK-NEXT:     %3 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<!llvm.struct<(memref<?xf32>, i32)>>) -> !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<memref<?xf32>>
// CHECK-NEXT:     %5 = affine.load %4[0] : memref<?xf32>
// CHECK-NEXT:     %6 = arith.mulf %5, %2 : f32
// CHECK-NEXT:     affine.store %6, %4[0] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
