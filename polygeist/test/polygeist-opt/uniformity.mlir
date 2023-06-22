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

// CHECK: test1_add1, uniformity: unknown
// CHECK: test1_add2, uniformity: non-uniform
// CHECK: test1_add3, uniformity: non-uniform
// CHECK: test1_add4, uniformity: uniform
func.func @test1(%arg0 : i1, %arg1: memref<?x!sycl_nd_item_2>)  {
  %c0_i32 = arith.constant 0 : i32  
  %true = arith.constant 1 : i1
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 2 : i64
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  // %arg0 uniformity is unknown -> %v0 uniformity also unknown

  %v0 = scf.if %arg0 -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  }
  %c1 = arith.constant 1 : i64  
  %add1 = arith.addi %v0, %c1 {tag = "test1_add1"} : i64

  // branch condition is non-uniform -> %v2 is non-uniform
  %cond2 = arith.cmpi slt, %tx, %c1 : i64
  %v2 = scf.if %cond2 -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  }
  %add2 = arith.addi %v2, %c1 {tag = "test1_add2"} : i64  

  // branch condition is uniform, but yielded value is non-uniform -> %v0 is non-uniform
  %v3 = scf.if %true -> i64 {
    scf.yield %tx : i64
  } else {
    scf.yield %c3 : i64
  }  
  %add3 = arith.addi %v3, %c1 {tag = "test1_add3"} : i64

  // branch condition is uniform, and yielded values is uniform -> %v0 is uniform
  %v4 = scf.if %true -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  }
  %add4 = arith.addi %v4, %c1 {tag = "test1_add4"} : i64

  return
}

