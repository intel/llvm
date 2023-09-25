// RUN: polygeist-opt -split-input-file -test-uniformity %s 2>&1 | FileCheck %s

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

// COM: Check uniformity of values yielded by branch operations.
func.func @test1a(%arg0 : i1, %arg1: memref<?x!sycl_nd_item_2>)  {
  %c0_i32 = arith.constant 0 : i32  
  %true = arith.constant 1 : i1
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64

  // COM: %arg0 uniformity is unknown -> result uniformity also unknown.
  // CHECK: test1a_v1, uniformity: unknown
  %v1 = scf.if %arg0 -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1a_v1"} 

  // COM: Branch condition is non-uniform -> result is non-uniform.
  // CHECK: test1a_v2, uniformity: non-uniform  
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64  
  %cond2 = arith.cmpi slt, %tx, %c2 : i64
  %v2 = scf.if %cond2 -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1a_v2"}

  // COM: Branch condition is uniform, but yielded value is non-uniform -> result is non-uniform.
  // CHECK: test1a_v3, uniformity: non-uniform  
  %v3 = scf.if %true -> i64 {
    scf.yield %tx : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1a_v3"} 

  // COM: Branch condition is uniform, and yielded values is uniform -> result is uniform.
  // CHECK: test1a_v4, uniformity: uniform  
  %v4 = scf.if %true -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1a_v4"} 

  // COM: The loop IV of a loop with constant bounds and step is uniform.
  // CHECK: test1a_iv1, uniformity: uniform
  affine.for %ii = 0 to 256 {
    %iv1 = arith.index_cast %ii {tag = "test1a_iv1"} : index to i64
  }

  // COM: The loop IV of a loop with non-uniform upper bound is non-uniform.
  // CHECK: test1a_iv2, uniformity: non-uniform
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tx_index = arith.index_cast %tx : i64 to index
  scf.for %ii = %c0 to %tx_index step %c1 {
     %iv2 = arith.index_cast %ii {tag = "test1a_iv2"} : index to i64
  }

  // COM: The result yielded by a loop with non-uniform loop carried value is non-uniform.
  // CHECK: test1a_iv3, uniformity: uniform  
  // CHECK: test1a_v5, uniformity: non-uniform
  %c4 = arith.constant 4 : index
  %v5 = scf.for %ii = %c0 to %c4 step %c1 iter_args(%sum = %tx) -> i64 {
    %iv3 = arith.index_cast %ii {tag = "test1a_iv3"} : index to i64
    %next_sum = arith.addi %iv3, %sum : i64
    scf.yield %next_sum : i64
  } {tag = "test1a_v5"}

  // COM: The result yielded by a loop with uniform bounds and step and a uniform 
  // COM: loop carried value is uniform.
  // CHECK: test1a_v6, uniformity: uniform
  %v6 = scf.for %ii = %c0 to %c4 step %c1 iter_args(%sum = %c0) -> index {
    %next_sum = arith.addi %ii, %sum : index
    scf.yield %next_sum : index
  } {tag = "test1a_v6"}

  // COM: If a uniform loop carried value is added to a non-uniform value the result yielded
  // COM: by the loop is non-uniform.
  // CHECK: test1a_v7, uniformity: non-uniform  
  %v7 = scf.for %ii = %c0 to %c4 step %c1 iter_args(%sum = %c0) -> index {
    %next_sum = arith.addi %tx_index, %sum : index
    scf.yield %next_sum : index
  } {tag = "test1a_v7"}

  // COM: The result yielded by a loop with non-uniform (mapped) upper bound is non-uniform.
  // CHECK: test1a_iv4, uniformity: non-uniform  
  affine.for %ii = 0 to min affine_map<(d0) -> (d0 + 2, d0 + 2)>(%tx_index) {
    %iv4 = arith.index_cast %ii {tag = "test1a_iv4"} : index to i64
  }

  return
}

func.func @test1b(%arg0 : index, %arg1: memref<?x!sycl_nd_item_2>)  {
  %c0_i32 = arith.constant 0 : i32  
  %true = arith.constant 1 : i1
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  // COM: %arg0 uniformity is unknown -> result uniformity also unknown.
  // CHECK: test1b_v1, uniformity: unknown
  %v1 = affine.if affine_set<(d0) : (d0 >= 0)>(%arg0) -> index {
    affine.yield %c2 : index
  } else {
    affine.yield %c3 : index
  } {tag = "test1b_v1"} 


  // COM: Branch condition is non-uniform -> result is non-uniform.
  // CHECK: test1b_v2, uniformity: non-uniform  
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64  
  %tx_index = arith.index_cast %tx : i64 to index
  %v2 = affine.if affine_set<(d0, d1) : (d1 >= d0)>(%tx_index, %c2) -> index {
    affine.yield %c2 : index
  } else {
    affine.yield %c3 : index
  } {tag = "test1b_v2"}

  // COM: Branch condition is uniform, but yielded value is non-uniform -> result is non-uniform.
  // CHECK: test1b_v3, uniformity: non-uniform  
  %v3 = affine.if affine_set<() : (0 >= 0)>() -> index {
    affine.yield %tx_index : index
  } else {
    affine.yield %c3 : index
  } {tag = "test1b_v3"}

  // COM: Branch condition is uniform, and yielded values is uniform -> result is uniform.
  // CHECK: test1b_v4, uniformity: uniform  
  %v4 = affine.if affine_set<() : (0 >= 0)>() -> index {
    affine.yield %c2 : index
  } else {
    affine.yield %c3 : index
  } {tag = "test1b_v4"} 

  return
}

// -----

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

// COM: Check the uniformity for operations that load from memory.
func.func @test2(%cond: i1, %val: i64, %arg1: memref<?x!sycl_nd_item_2>)  {
  %alloca = memref.alloca() : memref<10xi64>

  // COM: load memref op. is uniform, index is uniform -> result is uniform.
  // CHECK: test2_load1, uniformity: uniform  
  %c0 = arith.constant 0 : index    
  %load1 = memref.load %alloca[%c0] { tag = "test2_load1" } : memref<10xi64>

  // COM: load memref op. is uniform, index is non-uniform -> result is non-uniform.
  // CHECK: test2_load2, uniformity: non-uniform
  %c0_i32 = arith.constant 0 : i32  
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64
  %tx_cast = arith.index_cast %tx : i64 to index
  %load2 = memref.load %alloca[%tx_cast] { tag = "test2_load2" } : memref<10xi64>

  // COM: load memref op. uniformity is unknown due to store -> result uniformity is unknown.
  // CHECK: test2_load3, uniformity: unknown
  memref.store %val, %alloca[%c0]: memref<10xi64>
  %load3 = memref.load %alloca[%c0] { tag = "test2_load3" } : memref<10xi64>

  // COM: Store a non-uniform value in load memref operand -> result is non-uniform.
  // CHECK: test2_load4, uniformity: non-uniform
  memref.store %tx, %alloca[%c0]: memref<10xi64>  
  %load4 = memref.load %alloca[%c0] { tag = "test2_load4" } : memref<10xi64>  

  // COM: Although the stores in both branches kill the previous def. and store uniform values,
  // COM: the branch condition has unknown uniformity so it may be divergent. The load result 
  // COM: has unknown uniformity.
  scf.if %cond {
    %c1 = arith.constant 1 : i64
    memref.store %c1, %alloca[%c0] : memref<10xi64>
  } else {
    %c2 = arith.constant 2 : i64    
    memref.store %c2, %alloca[%c0] : memref<10xi64>
  }
  // CHECK: test2_load5, uniformity: unknown
  %load5 = memref.load %alloca[%c0] { tag = "test2_load5" } : memref<10xi64>

  // COM: the branch condition is non-uniform, so the value yielded by the scf.if is also non-uniform.
  %c0_i64 = arith.constant 0 : i64
  %cond1 = arith.cmpi sgt, %tx, %c0_i64 : i64
  %alloca1 = memref.alloca() : memref<10xi64> 
  %alloca2 = scf.if %cond1 -> memref<10xi64> {
    scf.yield %alloca1 : memref<10xi64>
  } else {
    %alloca2a = memref.alloca() : memref<10xi64>
    scf.yield %alloca2a : memref<10xi64>
  }
  // CHECK: test2_load6, uniformity: non-uniform
  %load6 = memref.load %alloca2[%c0] { tag = "test2_load6" } : memref<10xi64>

  // COM: Store a uniform value (%c4) through a memref (%alloca2) that is non-uniform and load it back 
  // COM: via an another memref (%alloca1) which in itself is uniform. The result should be non-uniform 
  // COM: because the potential reaching def. of %alloca1 (the store operation) is storing through a 
  // COM: non-uniform value.
  %c4 = arith.constant 4 : i64
  memref.store %c4, %alloca2[%c0] : memref<10xi64>
  // CHECK: test2_load7, uniformity: non-uniform
  %load7 = memref.load %alloca1[%c0] { tag = "test2_load7" } : memref<10xi64>

  return
}

// -----

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

gpu.module @device_func {

// COM: Check the uniformity inter-procedurally.
func.func private @test3(%cond: i1, %uniform_val: i64, %non_uniform_val : i64)  {
  %alloca = memref.alloca() : memref<10xi64>

  // COM: The stored value is known inter-procedurally to be uniform.
  // CHECK: test3_load1, uniformity: uniform
  %c0 = arith.constant 0 : index
  memref.store %uniform_val, %alloca[%c0]: memref<10xi64>
  %load1 = memref.load %alloca[%c0] { tag = "test3_load1" } : memref<10xi64>

  // COM: The condition is known inter-procedurally to be uniform, uniform value is stored.
  // CHECK: test3_load2, uniformity: uniform  
  scf.if %cond {
    memref.store %uniform_val, %alloca[%c0] : memref<10xi64>    
  } else {
    scf.yield
  }
  %load2 = memref.load %alloca[%c0] { tag = "test3_load2" } : memref<10xi64>

  // COM: The stored value is known inter-procedurally to be non-uniform.
  // CHECK: test3_load3, uniformity: non-uniform
  memref.store %non_uniform_val, %alloca[%c0]: memref<10xi64>
  %load3 = memref.load %alloca[%c0] { tag = "test3_load3" } : memref<10xi64>

  // COM: The condition is uniform but the store only partially kills the previous def. 
  // CHECK: test3_load4, uniformity: non-uniform  
  scf.if %cond {
    memref.store %uniform_val, %alloca[%c0] : memref<10xi64>
  } else {
    scf.yield
  }
  %load4 = memref.load %alloca[%c0] { tag = "test3_load4" } : memref<10xi64>  

  return
}

gpu.func @kernel(%cond: i1, %arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) kernel {
  %c0_i32 = arith.constant 0 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  %c1_i64 = arith.constant 1 : i64
  func.call @test3(%cond, %c1_i64, %tx) : (i1, i64, i64) -> ()
  gpu.return
}

}

// -----

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

gpu.module @device_func {

// COM: Check the uniformity inter-procedurally.
func.func private @test4(%alloca_cond: memref<1xi1>, %uniform_val: i64)  {
  %alloca = memref.alloca() : memref<10xi64>
  %c0 = arith.constant 0 : index  

  // COM: The loaded (load1) value was stored with a uniform value in the caller, so is uniform.
  // COM: The loaded (load2) value is also uniform because the reaching def. stored a uniform value 
  // COM: and the branch condition is uniform.
  // CHECK: test4_load1, uniformity: uniform    
  // CHECK: test4_load2, uniformity: uniform  
  %cond = memref.load %alloca_cond[%c0] { tag = "test4_load1" } : memref<1xi1>
  scf.if %cond {
    memref.store %uniform_val, %alloca[%c0] : memref<10xi64>    
  } else {
    scf.yield
  }
  %load2 = memref.load %alloca[%c0] { tag = "test4_load2" } : memref<10xi64>

  return
}

gpu.func @kernel(%cond: i1, %arg1: memref<?x!sycl_nd_item_2>) kernel {
  %c1_i64 = arith.constant 1 : i64
  %alloca = memref.alloca() : memref<1xi1>
  %c0 = arith.constant 0 : index

  // COM: Store the condition (uniform) into memory.
  memref.store %cond, %alloca[%c0] : memref<1xi1>
  func.call @test4(%alloca, %c1_i64) : (memref<1xi1>, i64) -> ()
  gpu.return
}

}

// -----

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

gpu.module @device_func {

func.func private @test5(%alloca_cond: memref<1xi1>)  {
  %alloca = memref.alloca() : memref<10xi64>
  %c0 = arith.constant 0 : index

  // FIXME: Improve uniformity analysis to handle multiple callers,
  //        with multiple underlying values.
  // CHECK: test5_load1, uniformity: unknown
  %cond = memref.load %alloca_cond[%c0] { tag = "test5_load1" } : memref<1xi1>

  return
}

gpu.func @kernel1(%cond: i1, %arg1: memref<?x!sycl_nd_item_2>) kernel {
  %alloca = memref.alloca() : memref<1xi1>
  %c0 = arith.constant 0 : index

  // COM: Store the condition (uniform) into memory.
  memref.store %cond, %alloca[%c0] : memref<1xi1>
  func.call @test5(%alloca) : (memref<1xi1>) -> ()
  gpu.return
}

gpu.func @kernel2(%cond: i1, %arg1: memref<?x!sycl_nd_item_2>) kernel {
  %alloca = memref.alloca() : memref<1xi1>
  %c0 = arith.constant 0 : index

  // COM: Store the condition (uniform) into memory.
  memref.store %cond, %alloca[%c0] : memref<1xi1>
  func.call @test5(%alloca) : (memref<1xi1>) -> ()
  gpu.return
}

}
