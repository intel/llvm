// RUN: mlir-opt -allow-unregistered-dialect %s -affine-scalrep | FileCheck %s

// Affine scalar replacement pass is a func::FuncOp pass.
// Ensure that simple_store_load function nested in gpu.module can be transformed by -affine-scalrep.

gpu.module @functions {
  // CHECK-LABEL: func @simple_store_load() {
  func.func @simple_store_load() {
    %cf7 = arith.constant 7.0 : f32
    %m = memref.alloc() : memref<10xf32>
    affine.for %i0 = 0 to 10 {
      affine.store %cf7, %m[%i0] : memref<10xf32>
      %v0 = affine.load %m[%i0] : memref<10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
    memref.dealloc %m : memref<10xf32>
    return
  // CHECK:       %[[C7:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    arith.addf %[[C7]], %[[C7]] : f32
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  }
}
