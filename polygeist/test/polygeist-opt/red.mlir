// RUN: polygeist-opt --detect-reduction %s | FileCheck %s

module  {
  // COM: Ensure array reductions are detected on SCF for loops.
  func.func @detect_reduction_scf_for_1(%arg0: memref<?xf32>) -> f32 {
    // CHECK-LABEL:   func.func @detect_reduction_scf_for_1
    // CHECK-DAG:       %cst = arith.constant 1.000000e+00 : f32
    // CHECK-DAG:       %cst_0 = arith.constant 0.000000e+00 : f32
    // CHECK-DAG:       %c0 = arith.constant 0 : index
    // CHECK-DAG:       %c1 = arith.constant 1 : index
    // CHECK-NEXT:      %dim = memref.dim %arg0, %c0 : memref<?xf32>
    // CHECK-NEXT:      %alloca = memref.alloca() : memref<1xf32>
    // CHECK-NEXT:      affine.store %cst_0, %alloca[0] : memref<1xf32>
    // CHECK-NEXT:      %alloca_1 = memref.alloca() : memref<1xf32>
    // CHECK-NEXT:      affine.store %cst, %alloca_1[0] : memref<1xf32>
    // CHECK-NEXT:      %0 = affine.load %alloca_1[0] : memref<1xf32>
    // CHECK-NEXT:      %1 = affine.load %alloca[0] : memref<1xf32>
    // CHECK-NEXT:      %2:2 = scf.for %arg1 = %c0 to %dim step %c1 iter_args(%arg2 = %0, %arg3 = %1) -> (f32, f32) {
    // CHECK-NEXT:        %6 = memref.load %arg0[%arg1] : memref<?xf32>
    // CHECK-NEXT:        %7 = arith.addf %arg3, %6 : f32
    // CHECK-NEXT:        %8 = arith.mulf %arg2, %6 : f32
    // CHECK-NEXT:        scf.yield %8, %7 : f32, f32
    // CHECK-NEXT:      }
    // CHECK-NEXT:      affine.store %2#1, %alloca[0] : memref<1xf32>
    // CHECK-NEXT:      affine.store %2#0, %alloca_1[0] : memref<1xf32>
    // CHECK-NEXT:      %3 = affine.load %alloca_1[0] : memref<1xf32>
    // CHECK-NEXT:      %4 = affine.load %alloca[0] : memref<1xf32>
    // CHECK-NEXT:      %5 = arith.addf %4, %3 : f32
    // CHECK-NEXT:      return %5 : f32
    // CHECK-NEXT:    }

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %arg0, %c0 : memref<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %1 = memref.alloca() : memref<1xf32>
    affine.store %cst, %1[0] : memref<1xf32>
    %2 = memref.alloca() : memref<1xf32>
    affine.store %cst_0, %2[0] : memref<1xf32>
    scf.for %arg1 = %c0 to %0 step %c1 {
      %6 = affine.load %2[0] : memref<1xf32>
      %7 = affine.load %1[0] : memref<1xf32>
      %8 = memref.load %arg0[%arg1] : memref<?xf32>
      %9 = arith.addf %7, %8 : f32
      %10 = arith.mulf %6, %8 : f32
      affine.store %9, %1[0] : memref<1xf32>
      affine.store %10, %2[0] : memref<1xf32>
    }
    %3 = affine.load %2[0] : memref<1xf32>
    %4 = affine.load %1[0] : memref<1xf32>
    %5 = arith.addf %4, %3 : f32
    return %5 : f32
  }

  // COM: Ensure array reductions are detected affine for loops.
  func.func @detect_reduction_affine_for_1(%arg0: memref<?xf32>) -> f32 {
    // CHECK-LABEL:   func.func @detect_reduction_affine_for_1
    // CHECK-DAG:       %cst = arith.constant 1.000000e+00 : f32
    // CHECK-DAG:       %cst_0 = arith.constant 0.000000e+00 : f32
    // CHECK-DAG:       %c0 = arith.constant 0 : index    
    // CHECK-NEXT:      %dim = memref.dim %arg0, %c0 : memref<?xf32>
    // CHECK-NEXT:      %alloca = memref.alloca() : memref<1xf32>
    // CHECK-NEXT:      affine.store %cst_0, %alloca[0] : memref<1xf32>
    // CHECK-NEXT:      %alloca_1 = memref.alloca() : memref<1xf32>
    // CHECK-NEXT:      affine.store %cst, %alloca_1[0] : memref<1xf32>
    // CHECK-NEXT:      %0 = affine.load %alloca_1[0] : memref<1xf32>
    // CHECK-NEXT:      %1 = affine.load %alloca[0] : memref<1xf32>
    // CHECK-NEXT:      %2:2 = affine.for %arg1 = 0 to %dim iter_args(%arg2 = %0, %arg3 = %1) -> (f32, f32) {
    // CHECK-NEXT:        %6 = affine.load %arg0[%arg1] : memref<?xf32>
    // CHECK-NEXT:        %7 = arith.addf %arg3, %6 : f32
    // CHECK-NEXT:        %8 = arith.mulf %arg2, %6 : f32
    // CHECK-NEXT:        affine.yield %8, %7 : f32, f32
    // CHECK-NEXT:      }
    // CHECK-NEXT:      affine.store %2#1, %alloca[0] : memref<1xf32>
    // CHECK-NEXT:      affine.store %2#0, %alloca_1[0] : memref<1xf32>
    // CHECK-NEXT:      %3 = affine.load %alloca_1[0] : memref<1xf32>
    // CHECK-NEXT:      %4 = affine.load %alloca[0] : memref<1xf32>
    // CHECK-NEXT:      %5 = arith.addf %4, %3 : f32
    // CHECK-NEXT:      return %5 : f32
    // CHECK-NEXT:    }

    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %1 = memref.alloca() : memref<1xf32>
    affine.store %cst, %1[0] : memref<1xf32>
    %2 = memref.alloca() : memref<1xf32>
    affine.store %cst_0, %2[0] : memref<1xf32>
    affine.for %arg1 = 0 to %0 {
      %6 = affine.load %2[0] : memref<1xf32>
      %7 = affine.load %1[0] : memref<1xf32>
      %8 = affine.load %arg0[%arg1] : memref<?xf32>
      %9 = arith.addf %7, %8 : f32
      %10 = arith.mulf %6, %8 : f32
      affine.store %9, %1[0] : memref<1xf32>
      affine.store %10, %2[0] : memref<1xf32>
    }
    %3 = affine.load %2[0] : memref<1xf32>
    %4 = affine.load %1[0] : memref<1xf32>
    %5 = arith.addf %4, %3 : f32
    return %5 : f32
  }

  // COM: Ensure reduction is not detected (index is not loop invariant).
  func.func @no_detect_reduction_affine_for_1(%arg0: memref<?xi32>)  {
    // CHECK-LABEL:   func.func @no_detect_reduction_affine_for_1
    // CHECK:         affine.for %arg1 = 0 to 8 {
    // CHECK-NEXT:      %0 = affine.load %arg0[%arg1] : memref<?xi32>
    // CHECK-NEXT:      %1 = arith.addi %0, {{.*}} : i32
    // CHECK-NEXT:      affine.store %1, %arg0[%arg1] : memref<?xi32>
    // CHECK-NEXT:    }
    
    %cst = arith.constant 4 : i32
    affine.for %arg1 = 0 to 8 {
      %0 = affine.load %arg0[%arg1] : memref<?xi32>
      %1 = arith.addi %0, %cst : i32
      affine.store %1, %arg0[%arg1] : memref<?xi32>
    }  
    return
  }      

  // COM: Ensure reduction is not detected (operand is not loop invariant).
  func.func @no_detect_reduction_affine_for_2(%arg0: memref<1xmemref<?xi32>>)  {
    // CHECK-LABEL:   func.func @no_detect_reduction_affine_for_2
    // CHECK:         affine.for %arg1 = 0 to 8 {
    // CHECK-NEXT:      %0 = affine.load %arg0[0] : memref<1xmemref<?xi32>>      
    // CHECK-NEXT:      %1 = affine.load %0[0] : memref<?xi32>
    // CHECK-NEXT:      %2 = arith.addi %1, {{.*}} : i32
    // CHECK-NEXT:      affine.store %2, %0[0] : memref<?xi32>
    // CHECK-NEXT:    }    

    %cst = arith.constant 4 : i32
    affine.for %arg1 = 0 to 8 {
      %0 = affine.load %arg0[0] : memref<1xmemref<?xi32>>
      %1 = affine.load %0[0] : memref<?xi32>
      %2 = arith.addi %1, %cst : i32
      affine.store %2, %0[0] : memref<?xi32>
    }  
    return
  }
}
