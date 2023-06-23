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

func.func @test1(%arg0 : i1, %arg1: memref<?x!sycl_nd_item_2>)  {
  %c0_i32 = arith.constant 0 : i32  
  %true = arith.constant 1 : i1
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 2 : i64
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  // %arg0 uniformity is unknown -> result uniformity also unknown.
  // CHECK: test1_v1, uniformity: unknown
  %v1 = scf.if %arg0 -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1_v1"} 

  // branch condition is non-uniform -> result is non-uniform.
  // CHECK: test1_v2, uniformity: non-uniform  
  %cond2 = arith.cmpi slt, %tx, %c2 : i64
  %v2 = scf.if %cond2 -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1_v2"}

  // branch condition is uniform, but yielded value is non-uniform -> result is non-uniform.
  // CHECK: test1_v3, uniformity: non-uniform  
  %v3 = scf.if %true -> i64 {
    scf.yield %tx : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1_v3"} 

  // branch condition is uniform, and yielded values is uniform -> result is uniform.
  // CHECK: test1_v4, uniformity: uniform  
  %v4 = scf.if %true -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  } {tag = "test1_v4"} 

  return
}

