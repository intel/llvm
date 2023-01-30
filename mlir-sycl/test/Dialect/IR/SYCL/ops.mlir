// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL: test_num_work_items
func.func @test_num_work_items() -> !sycl_range_1_ {
  %0 = sycl.num_work_items() : () -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// CHECK-LABEL: test_num_work_items_dim
func.func @test_num_work_items_dim(%i: i32) -> index {
  %0 = sycl.num_work_items(%i) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_global_id
func.func @test_global_id() -> !sycl_id_2_ {
  %0 = sycl.global_id() : () -> !sycl_id_2_
  return %0 : !sycl_id_2_
}

// CHECK-LABEL: test_global_id_dim
func.func @test_global_id_dim(%i: i32) -> index {
  %0 = sycl.global_id(%i) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_local_id
func.func @test_local_id() -> !sycl_id_3_ {
  %0 = sycl.local_id() : () -> !sycl_id_3_
  return %0 : !sycl_id_3_
}

// CHECK-LABEL: test_local_id_dim
func.func @test_local_id_dim(%i: i32) -> index {
  %0 = sycl.local_id(%i) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_global_offset
func.func @test_global_offset() -> !sycl_id_1_ {
  %0 = sycl.global_offset() : () -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// CHECK-LABEL: test_global_offset_dim
func.func @test_global_offset_dim(%i: i32) -> index {
  %0 = sycl.global_offset(%i) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_num_work_groups
func.func @test_num_work_groups() -> !sycl_range_2_ {
  %0 = sycl.num_work_groups() : () -> !sycl_range_2_
  return %0 : !sycl_range_2_
}

// CHECK-LABEL: test_num_work_groups_dim
func.func @test_num_work_groups_dim(%i: i32) -> index {
  %0 = sycl.num_work_groups(%i) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_work_group_size
func.func @test_work_group_size() -> !sycl_range_3_ {
  %0 = sycl.work_group_size() : () -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// CHECK-LABEL: test_work_group_size_dim
func.func @test_work_group_size_dim(%i: i32) -> index {
  %0 = sycl.work_group_size(%i) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_work_group_id
func.func @test_work_group_id() -> !sycl_id_1_ {
  %0 = sycl.work_group_id() : () -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// CHECK-LABEL: test_work_group_id_dim
func.func @test_work_group_id_dim(%i: i32) -> index {
  %0 = sycl.work_group_id(%i) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_num_sub_groups
func.func @test_num_sub_groups() -> i32 {
  %0 = sycl.num_sub_groups : () -> i32
  return %0 : i32
}

// CHECK-LABEL: test_sub_group_max_size
func.func @test_sub_group_max_size() -> i32 {
  %0 = sycl.sub_group_max_size : () -> i32
  return %0 : i32
}

// CHECK-LABEL: test_sub_group_size
func.func @test_sub_group_size() -> i32 {
  %0 = sycl.sub_group_size : () -> i32
  return %0 : i32
}

// CHECK-LABEL: test_sub_group_id
func.func @test_sub_group_id() -> i32 {
  %0 = sycl.sub_group_id : () -> i32
  return %0 : i32
}

// CHECK-LABEL: test_sub_group_local_id
func.func @test_sub_group_local_id() -> i32 {
  %0 = sycl.sub_group_local_id : () -> i32
  return %0 : i32
}

// CHECK-LABEL: test_num_work_items_const
func.func @test_num_work_items_const() -> index {
  %c0_i32 = arith.constant 0 : i32
  %0 = sycl.num_work_items(%c0_i32) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_global_id_const
func.func @test_global_id_const() -> index {
  %c1_i32 = arith.constant 1 : i32
  %0 = sycl.global_id(%c1_i32) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_local_id_const
func.func @test_local_id_const() -> index {
  %c2_i32 = arith.constant 2 : i32
  %0 = sycl.local_id(%c2_i32) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_global_offset_const
func.func @test_global_offset_const() -> index {
  %c0_i32 = arith.constant 0 : i32
  %0 = sycl.global_offset(%c0_i32) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_num_work_groups_const
func.func @test_num_work_groups_const() -> index {
  %c1_i32 = arith.constant 1 : i32
  %0 = sycl.num_work_groups(%c1_i32) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_work_group_size_const
func.func @test_work_group_size_const() -> index {
  %c2_i32 = arith.constant 2 : i32
  %0 = sycl.work_group_size(%c2_i32) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: test_work_group_id_const
func.func @test_work_group_id_const() -> index {
  %c0_i32 = arith.constant 0 : i32
  %0 = sycl.work_group_id(%c0_i32) : (i32) -> index
  return %0 : index
}
