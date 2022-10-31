// RUN: polygeist-opt --cpuify="method=distribute" --canonicalize --split-input-file %s | FileCheck %s

module {
  func.func private @use(%arg : i1)
  func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: memref<?xf32>, %len : index, %f : f32, %start : i1, %end : i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
        affine.parallel (%arg15, %arg16) = (0, 0) to (16, 16) {
            %r = scf.for %arg17 = %c0 to %len step %c1 iter_args(%mid = %start) -> (i1) {
              affine.store %f, %arg0[%arg15] : memref<?xf32>
              "polygeist.barrier"(%arg15, %arg16, %c0) : (index, index, index) -> ()
              scf.yield %end : i1
            }
            func.call @use(%r) : (i1) -> ()
        }
    return 
  }
}

// CHECK:   func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: memref<?xf32>, %arg1: index, %arg2: f32, %arg3: i1, %arg4: i1) {
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %alloca = memref.alloca() : memref<16x16xi1>
// CHECK-NEXT:     affine.parallel (%arg5, %arg6) = (0, 0) to (16, 16) {
// CHECK-NEXT:       memref.store %arg3, %alloca[%arg5, %arg6] : memref<16x16xi1>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.for %arg5 = %c0 to %arg1 step %c1 {
// CHECK-NEXT:       affine.parallel (%arg6, %arg7) = (0, 0) to (16, 16) {
// CHECK-NEXT:         affine.store %arg2, %arg0[%arg6] : memref<?xf32>
// CHECK-NEXT:         memref.store %arg4, %alloca[%arg6, %arg7] : memref<16x16xi1>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.parallel (%arg5, %arg6) = (0, 0) to (16, 16) {
// CHECK-NEXT:       %0 = memref.load %alloca[%arg5, %arg6] : memref<16x16xi1>
// CHECK-NEXT:       func.call @use(%0) : (i1) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
