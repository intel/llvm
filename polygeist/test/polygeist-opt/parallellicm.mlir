// RUN: polygeist-opt --parallel-licm --split-input-file --mlir-pass-statistics %s 2>&1 | FileCheck %s

module {
func.func private @use(f32) 
// COM: Ensure loop invariant load is hoisted.
func.func @scf_parallel_hoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK: ParallelLICM
  // CHECK:  (S) 1 num-operations-hoisted - Number of operations hoisted

  // CHECK:       func.func @scf_parallel_hoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
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
}

// -----

module {
func.func private @use(f32) 
// COM:  Ensure loop invariant store + load are both hoisted.
func.func @scf_parallel_hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK:       func.func @scf_parallel_hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
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
}

// -----

module {
func.func private @use(f32) 
func.func private @get() -> (f32) 
// COM: Ensure store with loop variant operand is not hoisted, and that subsequent aliased load is not hoisted.
func.func @scf_parallel_nohoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK:       func.func @scf_parallel_nohoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK-NEXT:     %c1 = arith.constant 1 : index
  // CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
  // CHECK-NEXT:     scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
  // CHECK-NEXT:       %0 = func.call @get() : () -> f32
  // CHECK-NEXT:       memref.store %0, %alloca[] : memref<f32>
  // CHECK-NEXT:       %1 = memref.load %alloca[] : memref<f32>
  // CHECK-NEXT:       func.call @use(%1) : (f32) -> ()
  // CHECK-NEXT:       scf.yield
  // CHECK-NEXT:     }

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

// -----
module {
func.func private @use(f32) 
// COM: Ensure loop invariant load is hoisted.
func.func @affine_parallel_hoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %arg3 : index, %arg4 : index, %arg5: index, %arg6: index) {
  // CHECK:       func.func @affine_parallel_hoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) {
  // CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
  // CHECK-NEXT:     memref.store %cst, %alloca[] : memref<f32>
  // CHECK-NEXT:     affine.if #set(%arg1, %arg2, %arg5, %arg3, %arg4, %arg6) {
  // CHECK-NEXT:       %0 = memref.load %alloca[] : memref<f32>
  // CHECK-NEXT:       affine.parallel (%arg7, %arg8) = (max(%arg1, %arg2), %arg5) to (min(%arg3, %arg4), %arg6) {
  // CHECK-NEXT:         func.call @use(%0) : (f32) -> ()
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }

  %cst = arith.constant 0.000000e+00 : f32
  %a = memref.alloca() : memref<f32>
  memref.store %cst, %a[] : memref<f32>
  affine.parallel (%arg7, %arg8) = (max(%arg1, %arg2), %arg5) to (min(%arg3, %arg4), %arg6) {
    %v = memref.load %a[] : memref<f32>
    func.call @use(%v) : (f32) -> ()
  }
  return
}
}

// -----

module {
func.func private @use(f32) 
// COM: Ensure unaliased loop invariant load is hoisted, and reduction (load, op, store) is not hoisted.
func.func @affine_for_hoist1(%arg0: memref<?xf32>, %arg1: f32) {
  // CHECK:       #set = affine_set<() : (9 >= 0)>
  // CHECK:       func.func @affine_for_hoist1(%arg0: memref<?xf32>, %arg1: f32) {
  // CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
  // CHECK-NEXT:     memref.store %arg1, %alloca[] : memref<f32>
  // CHECK-NEXT:     affine.if #set() {
  // CHECK-NEXT:       %0 = memref.load %alloca[] : memref<f32>
  // CHECK-NEXT:       affine.for %arg2 = 0 to 10 {
  // CHECK-NEXT:         %1 = affine.load %arg0[0] : memref<?xf32>
  // CHECK-NEXT:         %2 = arith.addf %1, %0 : f32
  // CHECK-NEXT:         affine.store %2, %arg0[0] : memref<?xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }

  %a = memref.alloca() : memref<f32>
  memref.store %arg1, %a[] : memref<f32>
  affine.for %arg2 = 0 to 10 {    
    %tmp = memref.load %a[] : memref<f32>
    %arr = affine.load %arg0[0] : memref<?xf32>
    %add = arith.addf %arr, %tmp : f32
    affine.store %add, %arg0[0] : memref<?xf32>
  }
  return
}
}

// -----

module {
// COM: Ensure reductions loops guards are correct.  
func.func @affine_for_hoist2(%arg0: memref<?xi32>, %arg1: i32) -> (i32) {
  // CHECK:       #set = affine_set<() : (9 >= 0)>
  // CHECK:        func.func @affine_for_hoist2(%arg0: memref<?xi32>, %arg1: i32) -> i32 {
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<1xi32>
  // CHECK-NEXT:    %c3_i32 = arith.constant 3 : i32
  // CHECK-NEXT:    %0 = affine.if #set() -> i32 {
  // CHECK-NEXT:      %1 = affine.load %alloca[0] : memref<1xi32>
  // CHECK-NEXT:      %2 = affine.for %arg2 = 0 to 10 iter_args(%arg3 = %c3_i32) -> (i32) {
  // CHECK-NEXT:        %3 = arith.addi %arg3, %1 : i32
  // CHECK-NEXT:        affine.yield %3 : i32
  // CHECK-NEXT:      }
  // CHECK-NEXT:      affine.yield %2 : i32
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      affine.yield %c3_i32 : i32
  // CHECK-NEXT:    }
  // CHECK-NEXT:    return %0 : i32

  %alloca = memref.alloca() : memref<1xi32>    
  %sum0 = arith.constant 3 : i32    
  %sum = affine.for %arg2 = 0 to 10 iter_args(%sum_iter = %sum0) -> (i32) {    
    %tmp = affine.load %alloca[0] : memref<1xi32>
    %sum_next = arith.addi %sum_iter, %tmp : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}
}

// -----
