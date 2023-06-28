// RUN: polygeist-opt -split-input-file -test-uniformity %s 2>&1 | FileCheck %s

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_gb = !sycl.accessor<[2, f32, read, global_buffer], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

// COM: Check uniformity of values yielded by branch operations.
func.func @test1(%arg0 : i1, %arg1: memref<?x!sycl_nd_item_2>)  {
  %c0_i32 = arith.constant 0 : i32  
  %true = arith.constant 1 : i1
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64

  // COM: %arg0 uniformity is unknown -> result uniformity also unknown.
  // CHECK: test1_v1, uniformity: unknown
  %v1 = scf.if %arg0 -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1_v1"} 

  // COM: branch condition is non-uniform -> result is non-uniform.
  // CHECK: test1_v2, uniformity: non-uniform  
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64  
  %cond2 = arith.cmpi slt, %tx, %c2 : i64
  %v2 = scf.if %cond2 -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1_v2"}

  // COM: branch condition is uniform, but yielded value is non-uniform -> result is non-uniform.
  // CHECK: test1_v3, uniformity: non-uniform  
  %v3 = scf.if %true -> i64 {
    scf.yield %tx : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1_v3"} 

  // COM: branch condition is uniform, and yielded values is uniform -> result is uniform.
  // CHECK: test1_v4, uniformity: uniform  
  %v4 = scf.if %true -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1_v4"} 

  return
}

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
    memref.store %c1, %alloca[%c0]: memref<10xi64>
  } else {
    %c2 = arith.constant 2 : i64    
    memref.store %c2, %alloca[%c0]: memref<10xi64>
  }
  // CHECK: test2_load5, uniformity: unknown
  %load5 = memref.load %alloca[%c0] { tag = "test2_load5" } : memref<10xi64>

  // COM: the branch condition is non-uniform, so the value yielded by the scf.if is also non-uniform.
  %c0_i64 = arith.constant 0 : i64
  %cond1 = arith.cmpi sgt, %tx, %c0_i64 : i64
  %alloca1 = scf.if %cond1 -> memref<10xi64> {
    %alloca1 = memref.alloca() : memref<10xi64>
    scf.yield %alloca1 : memref<10xi64>
  } else {
    %alloca2 = memref.alloca() : memref<10xi64>
    scf.yield %alloca2 : memref<10xi64>    
  }
  // CHECK: test2_load6, uniformity: non-uniform
  %load6 = memref.load %alloca1[%c0] { tag = "test2_load6" } : memref<10xi64>

  return
}
