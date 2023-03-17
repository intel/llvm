// RUN: polygeist-opt --licm="relaxed-aliasing=false" --split-input-file %s 2>&1 | FileCheck %s
// RUN: polygeist-opt --licm="relaxed-aliasing=true" --split-input-file %s 2>&1 | FileCheck --check-prefix=CHECK-RELAXED-ALIASING %s

// COM: Test LICM on scf.for loops.
module {
// COM: Ensure loop invariant load/store is hoisted.
func.func @scf_for_hoist1(%arg0: memref<f32>, %arg1: index, %arg2: index) {
  // CHECK:       func.func @scf_for_hoist1(%arg0: memref<f32>, %arg1: index, %arg2: index) {
  // CHECK-DAG      %cst = arith.constant 2.000000e+00 : f32    
  // CHECK-DAG:     %c1 = arith.constant 1 : index
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<f32>
  // CHECK-NEXT:    memref.store %cst, %alloca[] : memref<f32>
  // CHECK-NEXT:    %0 = arith.cmpi slt, %arg1, %arg2 : index
  // CHECK-NEXT:    scf.if %0 {
  // CHECK-NEXT:      %1 = memref.load %alloca[] : memref<f32>
  // CHECK-NEXT:      memref.store %1, %arg0[] : memref<f32>  
  // CHECK-NEXT:      scf.for %arg3 = %arg1 to %arg2 step %c1 {
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }

  %cst = arith.constant 2.000000e+00 : f32
  %c1 = arith.constant 1 : index 
  %a = memref.alloca() : memref<f32>
  memref.store %cst, %a[] : memref<f32>
  scf.for %arg3 = %arg1 to %arg2 step %c1 {
    %v = memref.load %a[] : memref<f32>
    memref.store %v, %arg0[] : memref<f32>
  }
  return
}

// COM: Ensure unaliased loop invariant load is hoisted, and reduction (load, op, store) is not hoisted.
func.func @scf_for_hoist2(%arg0: memref<f32>, %arg1: index, %arg2: index, %arg3: f32) {
  // CHECK:       func.func @scf_for_hoist2(%arg0: memref<f32>, %arg1: index, %arg2: index, %arg3: f32) {
  // CHECK-NEXT:    %c1 = arith.constant 1 : index
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<f32>
  // CHECK-NEXT:    memref.store %arg3, %alloca[] : memref<f32>
  // CHECK-NEXT:    %0 = arith.cmpi slt, %arg1, %arg2 : index  
  // CHECK-NEXT:    scf.if %0 {
  // CHECK-NEXT:      %1 = memref.load %alloca[] : memref<f32>
  // CHECK-NEXT:      scf.for %arg4 = %arg1 to %arg2 step %c1 {
  // CHECK-NEXT:        %2 = memref.load %arg0[] : memref<f32>
  // CHECK-NEXT:        %3 = arith.addf %2, %1 : f32
  // CHECK-NEXT:        memref.store %3, %arg0[] : memref<f32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }

  %c1 = arith.constant 1 : index 
  %a = memref.alloca() : memref<f32>
  memref.store %arg3, %a[] : memref<f32>
  scf.for %arg4 = %arg1 to %arg2 step %c1 {
    %tmp = memref.load %a[] : memref<f32>
    %arr = memref.load %arg0[] : memref<f32>
    %add = arith.addf %arr, %tmp : f32
    memref.store %add, %arg0[] : memref<f32>
  }
  return
}

// COM: Ensure reductions loops guards are correct.  
func.func @scf_for_hoist3(%arg0: memref<f32>, %arg1: index, %arg2: index) -> (i32) {
  // CHECK:        func.func @scf_for_hoist3(%arg0: memref<f32>, %arg1: index, %arg2: index) -> i32 {
  // CHECK-NEXT:    %c1 = arith.constant 1 : index    
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<i32>
  // CHECK-NEXT:    %c3_i32 = arith.constant 3 : i32      
  // CHECK-NEXT:    %0 = arith.cmpi slt, %arg1, %arg2 : index  
  // CHECK-NEXT:    %1 = scf.if %0 -> (i32) {
  // CHECK-NEXT:      %2 = memref.load %alloca[] : memref<i32>
  // CHECK-NEXT:      %3 = scf.for %arg3 = %arg1 to %arg2 step %c1 iter_args(%arg4 = %c3_i32) -> (i32) {
  // CHECK-NEXT:        %4 = arith.addi %arg4, %2 : i32
  // CHECK-NEXT:        scf.yield %4 : i32
  // CHECK-NEXT:      }
  // CHECK-NEXT:      scf.yield %3 : i32
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      scf.yield %c3_i32 : i32
  // CHECK-NEXT:    }
  // CHECK-NEXT:    return %1 : i32

  %c1 = arith.constant 1 : index 
  %alloca = memref.alloca() : memref<i32>    
  %sum0 = arith.constant 3 : i32
  %sum = scf.for %arg3 = %arg1 to %arg2 step %c1 iter_args(%sum_iter = %sum0) -> (i32) {    
    %tmp = memref.load %alloca[] : memref<i32>
    %sum_next = arith.addi %sum_iter, %tmp : i32
    scf.yield %sum_next : i32
  }
  return %sum : i32
}
}

// -----

// COM: Test LICM on scf.parallel loops.
module {
func.func private @use(f32) 
func.func private @get() -> (f32) 

// COM: Ensure loop invariant load is hoisted.
func.func @scf_parallel_hoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK:       func.func @scf_parallel_hoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:     %c1 = arith.constant 1 : index
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<f32>
  // CHECK-NEXT:    memref.store %cst, %alloca[] : memref<f32>
  // CHECK-NEXT:    %0 = arith.cmpi slt, %arg1, %arg2 : index
  // CHECK-NEXT:    scf.if %0 {
  // CHECK-NEXT:      %1 = memref.load %alloca[] : memref<f32>
  // CHECK-NEXT:      scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
  // CHECK-NEXT:        func.call @use(%1) : (f32) -> ()
  // CHECK-NEXT:        scf.yield
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }

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

// COM:  Ensure loop invariant store + load are both hoisted.
func.func @scf_parallel_hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK:       func.func @scf_parallel_hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:     %c1 = arith.constant 1 : index
  // CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
  // CHECK-NEXT:     %0 = arith.cmpi slt, %arg1, %arg2 : index
  // CHECK-NEXT:     scf.if %0 {
  // CHECK-NEXT:       memref.store %cst, %alloca[] : memref<f32>
  // CHECK-NEXT:       %1 = memref.load %alloca[] : memref<f32>
  // CHECK-NEXT:       scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
  // CHECK-NEXT:         func.call @use(%1) : (f32) -> ()
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

// COM: Test LICM on affine.for loops.
!sycl_id_1 = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1 = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_f32_rw_gb = !sycl.accessor<[1, f32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1, !sycl_range_1, !sycl_range_1)>, !llvm.struct<(memref<?xf32, 1>)>)>

module {

// CHECK: #set = affine_set<()[s0, s1] : (s1 - s0 - 1 >= 0)>
// CHECK: #set1 = affine_set<() : (9 >= 0)>

// COM: Ensure loop invariant load and store are hoisted.
func.func @affine_for_hoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK:       func.func @affine_for_hoist1(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
  // CHECK-DAG      %cst = arith.constant 2.000000e+00 : f32    
  // CHECK-DAG:     %c1 = arith.constant 1 : index
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<2xf32>
  // CHECK-NEXT:    memref.store %cst, %alloca[%c1] : memref<2xf32>
  // CHECK-NEXT:    affine.if #set()[%arg1, %arg2] {
  // CHECK-NEXT:      %0 = affine.load %alloca[%c1] : memref<2xf32>
  // CHECK-NEXT:      affine.store %0, %arg0[%c1] : memref<?xf32>
  // CHECK-NEXT:      affine.for %arg3 = %arg1 to %arg2 {
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }

  %cst = arith.constant 2.000000e+00 : f32
  %c1 = arith.constant 1 : index 
  %a = memref.alloca() : memref<2xf32>
  memref.store %cst, %a[%c1] : memref<2xf32>
  affine.for %arg3 = %arg1 to %arg2 {
    %v = affine.load %a[%c1] : memref<2xf32>
    affine.store %v, %arg0[%c1] : memref<?xf32>
  }
  return
}

// COM: Ensure unaliased loop invariant load is hoisted, and reduction (load, op, store) is not hoisted.
func.func @affine_for_hoist2(%arg0: memref<?xf32>, %arg1: f32) {
  // CHECK:       func.func @affine_for_hoist2(%arg0: memref<?xf32>, %arg1: f32) {
  // CHECK-NEXT:     %alloca = memref.alloca() : memref<f32>
  // CHECK-NEXT:     memref.store %arg1, %alloca[] : memref<f32>
  // CHECK-NEXT:     affine.if #set1() {
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

// COM: Ensure reductions loops guards are correct.
func.func @affine_for_hoist3(%arg0: memref<?xi32>, %arg1: i32) -> (i32) {
  // CHECK:        func.func @affine_for_hoist3(%arg0: memref<?xi32>, %arg1: i32) -> i32 {
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<1xi32>
  // CHECK-NEXT:    %c3_i32 = arith.constant 3 : i32
  // CHECK-NEXT:    %0 = affine.if #set1() -> i32 {
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

// COM: Ensure aliased store dominating load can be hoisted.
func.func @affine_for_hoist4(%arg0: memref<?xi32>) {
  // CHECK:        func.func @affine_for_hoist4(%arg0: memref<?xi32>) {
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<1xi32>
  // CHECK-NEXT:    %c3_i32 = arith.constant 3 : i32
  // CHECK-NEXT:    affine.if #set1() { 
  // CHECK-NEXT:      affine.store %c3_i32, %alloca[0] : memref<1xi32>
  // CHECK-NEXT:      %0 = affine.load %alloca[0] : memref<1xi32>  
  // CHECK-NEXT:      affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:        %1 = affine.load %arg0[0] : memref<?xi32>
  // CHECK-NEXT:        %2 = arith.addi %1, %0 : i32
  // CHECK-NEXT:        affine.store %2, %arg0[0] : memref<?xi32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }

  %alloca = memref.alloca() : memref<1xi32>
  %c3 = arith.constant 3 : i32
  affine.for %arg1 = 0 to 10 {
    // Store can be hoisted because it is the only reaching definition for the first load. 
    //  - the store dominates the aliased load and 
    //  - there is no other aliased store in the loop
    affine.store %c3, %alloca[0] : memref<1xi32>
    %c3_1 = affine.load %alloca[0] : memref<1xi32>
    %arr = affine.load %arg0[0] : memref<?xi32>
    %add = arith.addi %arr, %c3_1 : i32
    affine.store %add, %arg0[0] : memref<?xi32>
  }
  return
}  

// COM: Ensure accessor.subscript operation is hoisted and the load + store instructions before the 
// COM: accessor.subscript operation are also hoisted (because %4 is not aliased with %alloca).
func.func @affine_for_hoist5(%arg0: memref<?x!sycl_accessor_1_f32_rw_gb, 4>) {
  // CHECK:        func.func @affine_for_hoist5(%arg0: memref<?x!sycl_accessor_1_f32_rw_gb, 4>) {
  // CHECK-DAG:      %alloca = memref.alloca() : memref<1x!sycl_id_1_>
  // CHECK-DAG:      %alloca_0 = memref.alloca() : memref<1x!sycl_id_1_>  
  // CHECK:          affine.if #set1() {
  // CHECK-NEXT:       %0 = affine.load %alloca[0] : memref<1x!sycl_id_1_>
  // CHECK-NEXT:       affine.store %0, %alloca_0[0] : memref<1x!sycl_id_1_>
  // CHECK-NEXT:       %1 = sycl.accessor.subscript %arg0[%alloca_0] {{.*}} : (memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<1x!sycl_id_1_>) -> memref<?xf32, 4>
  // CHECK-NEXT:       affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:         %2 = affine.load %1[0] : memref<?xf32, 4>
  // CHECK-NEXT:         %3 = arith.addf %2, {{.*}} : f32
  // CHECK-NEXT:         affine.store %3, %1[0] : memref<?xf32, 4>
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }    

  // CHECK-RELAXED-ALIASING:         func.func @affine_for_hoist5(%arg0: memref<?x!sycl_accessor_1_f32_rw_gb, 4>) {
  // CHECK-RELAXED-ALIASING-DAG:      %alloca = memref.alloca() : memref<1x!sycl_id_1_>
  // CHECK-RELAXED-ALIASING-DAG:      %alloca_0 = memref.alloca() : memref<1x!sycl_id_1_>  
  // CHECK-RELAXED-ALIASING:          sycl.constructor @id(%memspacecast, %c64_i64) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
  // CHECK-RELAXED-ALIASING-NEXT:     affine.for %arg1 = 0 to 10 {
  // CHECK-RELAXED-ALIASING-NEXT:      %0 = affine.load %alloca[0] : memref<1x!sycl_id_1_>
  // CHECK-RELAXED-ALIASING-NEXT:      affine.store %0, %alloca_0[0] : memref<1x!sycl_id_1_>
  // CHECK-RELAXED-ALIASING-NEXT:      %1 = sycl.accessor.subscript %arg0[%alloca_0] {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<1x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIfLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERfNS0_2idILi1EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<1x!sycl_id_1_>) -> memref<?xf32, 4>
  // CHECK-RELAXED-ALIASING-NEXT:      %2 = affine.load %1[0] : memref<?xf32, 4>
  // CHECK-RELAXED-ALIASING-NEXT:      %3 = arith.addf %2, %cst : f32
  // CHECK-RELAXED-ALIASING-NEXT:      affine.store %3, %1[0] : memref<?xf32, 4>
  // CHECK-RELAXED-ALIASING-NEXT:    }

  %alloca = memref.alloca() : memref<1x!sycl_id_1>  
  %alloca_0 = memref.alloca() : memref<1x!sycl_id_1>
  %c64_i64 = arith.constant 64 : i64
  %cst = arith.constant 1.000000e+01 : f32

  %0 = memref.cast %alloca : memref<1x!sycl_id_1> to memref<?x!sycl_id_1>
  %1 = memref.memory_space_cast %0 : memref<?x!sycl_id_1> to memref<?x!sycl_id_1, 4>
  sycl.constructor @id(%1, %c64_i64) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1, 4>, i64)

  affine.for %arg1 = 0 to 10 {    
    %2 = affine.load %alloca[0] : memref<1x!sycl_id_1>
    affine.store %2, %alloca_0[0] : memref<1x!sycl_id_1>
    %3 = sycl.accessor.subscript %arg0[%alloca_0] {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<1x!sycl_id_1>], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIfLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERfNS0_2idILi1EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<1x!sycl_id_1>) -> memref<?xf32, 4>
    %4 = affine.load %3[0] : memref<?xf32, 4>
    %5 = arith.addf %4, %cst : f32
    affine.store %5, %3[0] : memref<?xf32, 4>
  }
  return
}

// COM: Ensure aliased store after dominating load cannot be hoisted.
func.func @affine_for_nohoist1(%arg0: memref<?xi32>) {
  // CHECK:        func.func @affine_for_nohoist1(%arg0: memref<?xi32>) {
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<1xi32>
  // CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
  // CHECK-DAG:     %c4_i32 = arith.constant 4 : i32  
  // CHECK-NEXT:    affine.store %c3_i32, %alloca[0] : memref<1xi32>
  // CHECK-NEXT:    affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:      %0 = affine.load %alloca[0] : memref<1xi32> 
  // CHECK-NEXT:      affine.store %c4_i32, %alloca[0] : memref<1xi32>
  // CHECK-NEXT:      %1 = affine.load %arg0[0] : memref<?xi32>
  // CHECK-NEXT:      %2 = arith.addi %1, %0 : i32
  // CHECK-NEXT:      affine.store %2, %arg0[0] : memref<?xi32>
  // CHECK-NEXT:    }

  %alloca = memref.alloca() : memref<1xi32>    
  %c3 = arith.constant 3 : i32    
  %c4 = arith.constant 4 : i32      
  affine.store %c3, %alloca[0] : memref<1xi32>
  affine.for %arg2 = 0 to 10 {    
    // Cannot hoist the load because the loop has a store that can change the loaded result. 
    %c3_1 = affine.load %alloca[0] : memref<1xi32>
    // Cannot hoist the store because it changes the value loaded by the previous operation, 
    // (the store does not dominate the load %c3_1).
    affine.store %c4, %alloca[0] : memref<1xi32>
    %arr = affine.load %arg0[0] : memref<?xi32>
    %add = arith.addi %arr, %c3_1 : i32
    affine.store %add, %arg0[0] : memref<?xi32>
  }
  return
}
}

// -----

// COM: Test LICM on affine.parallel loops.
// CHECK: #set = affine_set<(d0, d1, d2, d3, d4, d5) : (d3 - d0 - 1 >= 0, d3 - d1 - 1 >= 0, d4 - d0 - 1 >= 0, d4 - d1 - 1 >= 0, d5 - d2 - 1 >= 0)>
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
