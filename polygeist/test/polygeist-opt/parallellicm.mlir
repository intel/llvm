// RUN: polygeist-opt --parallel-licm --split-input-file %s | FileCheck %s

module {
  func.func private @use(f32) 
  func.func @hoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index 
    %a = memref.alloca() : memref<f32>
    memref.store %cst, %a[] : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
  func.func @hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index 
    %a = memref.alloca() : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      memref.store %cst, %a[] : memref<f32>
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
  func.func private @get() -> (f32) 
  func.func @nohoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %c1 = arith.constant 1 : index 
    %a = memref.alloca() : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      %cst = func.call @get() : () -> (f32)
      memref.store %cst, %a[] : memref<f32>
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
}

// CHECK:   func.func @hoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
// CHECK-NEXT:     memref.store %cst, %alloca[] : memref<f32>
// CHECK-NEXT:     %0 = arith.addi %arg1, %c1 : index
// CHECK-NEXT:     %1 = arith.cmpi sle, %0, %arg2 : index
// CHECK-NEXT:     scf.if %1 {
// CHECK-NEXT:       %2 = memref.load %alloca[] : memref<f32>
// CHECK-NEXT:       scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
// CHECK-NEXT:         func.call @use(%2) : (f32) -> ()
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
// CHECK-NEXT:     %0 = arith.addi %arg1, %c1 : index
// CHECK-NEXT:     %1 = arith.cmpi sle, %0, %arg2 : index
// CHECK-NEXT:     scf.if %1 {
// CHECK-NEXT:       memref.store %cst, %alloca[] : memref<f32>
// CHECK-NEXT:       %2 = memref.load %alloca[] : memref<f32>
// CHECK-NEXT:       scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
// CHECK-NEXT:         func.call @use(%2) : (f32) -> ()
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @nohoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
// CHECK-NEXT:     scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
// CHECK-NEXT:       %0 = func.call @get() : () -> f32
// CHECK-NEXT:       memref.store %0, %alloca[] : memref<f32>
// CHECK-NEXT:       %1 = memref.load %alloca[] : memref<f32>
// CHECK-NEXT:       func.call @use(%1) : (f32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----

module {
  func.func private @use(f32) 
  func.func @affhoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %arg3 : index, %arg4 : index, %arg5: index, %arg6: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index 
    %a = memref.alloca() : memref<f32>
    memref.store %cst, %a[] : memref<f32>
    affine.parallel (%arg7, %arg8) = (max(%arg1, %arg2), %arg5) to (min(%arg3, %arg4), %arg6) {
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
}

// #set = affine_set<(d0, d1, d2, d3, d4, d5) : (d3 - d0 - 1 >= 0, d3 - d1 - 1 >= 0, d4 - d0 - 1 >= 0, d4 - d1 - 1 >= 0, d5 - d2 - 1 >= 0)>

// CHECK:   func.func @affhoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
// CHECK-NEXT:     memref.store %cst, %alloca[] : memref<f32>
// CHECK-NEXT:     affine.if #set(%arg1, %arg2, %arg5, %arg3, %arg4, %arg6) {
// CHECK-NEXT:       %0 = memref.load %alloca[] : memref<f32>
// CHECK-NEXT:       affine.parallel (%arg7, %arg8) = (max(%arg1, %arg2), %arg5) to (min(%arg3, %arg4), %arg6) {
// CHECK-NEXT:         func.call @use(%0) : (f32) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

