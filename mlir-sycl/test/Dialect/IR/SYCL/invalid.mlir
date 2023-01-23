// RUN: sycl-mlir-opt -split-input-file %s -verify-diagnostics

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

func.func @test_num_work_items(%i: i32) -> !sycl_range_1_ {
  // expected-error @+1 {{'sycl.num_work_items' op Expecting an index return value for this cardinality}}
  %0 = sycl.num_work_items(%i) : (i32) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

func.func @test_num_work_items_dim() -> index {
  // expected-error @+1 {{'sycl.num_work_items' op Not expecting an index return value for this cardinality}}
  %0 = sycl.num_work_items() : () -> index
  return %0 : index
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>

func.func @test_global_id(%i: i32) -> !sycl_id_2_ {
  // expected-error @+1 {{'sycl.global_id' op Expecting an index return value for this cardinality}}
  %0 = sycl.global_id(%i) : (i32) -> !sycl_id_2_
  return %0 : !sycl_id_2_
}

// -----

func.func @test_global_id_dim() -> index {
  // expected-error @+1 {{'sycl.global_id' op Not expecting an index return value for this cardinality}}
  %0 = sycl.global_id() : () -> index
  return %0 : index
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

func.func @test_local_id(%i: i32) -> !sycl_id_3_ {
  // expected-error @+1 {{'sycl.local_id' op Expecting an index return value for this cardinality}}
  %0 = sycl.local_id(%i) : (i32) -> !sycl_id_3_
  return %0 : !sycl_id_3_
}

// -----

func.func @test_local_id_dim() -> index {
  // expected-error @+1 {{'sycl.local_id' op Not expecting an index return value for this cardinality}}
  %0 = sycl.local_id() : () -> index
  return %0 : index
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

func.func @test_global_offset(%i: i32) -> !sycl_id_1_ {
  // expected-error @+1 {{'sycl.global_offset' op Expecting an index return value for this cardinality}}
  %0 = sycl.global_offset(%i) : (i32) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

func.func @test_global_offset_dim() -> index {
  // expected-error @+1 {{'sycl.global_offset' op Not expecting an index return value for this cardinality}}
  %0 = sycl.global_offset() : () -> index
  return %0 : index
}

// -----

!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>

func.func @test_num_work_groups(%i: i32) -> !sycl_range_2_ {
  // expected-error @+1 {{'sycl.num_work_groups' op Expecting an index return value for this cardinality}}
  %0 = sycl.num_work_groups(%i) : (i32) -> !sycl_range_2_
  return %0 : !sycl_range_2_
}

// -----

func.func @test_num_work_groups_dim() -> index {
  // expected-error @+1 {{'sycl.num_work_groups' op Not expecting an index return value for this cardinality}}
  %0 = sycl.num_work_groups() : () -> index
  return %0 : index
}

// -----

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

func.func @test_work_group_size(%i: i32) -> !sycl_range_3_ {
  // expected-error @+1 {{'sycl.work_group_size' op Expecting an index return value for this cardinality}}
  %0 = sycl.work_group_size(%i) : (i32) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

func.func @test_work_group_size_dim() -> index {
  // expected-error @+1 {{'sycl.work_group_size' op Not expecting an index return value for this cardinality}}
  %0 = sycl.work_group_size() : () -> index
  return %0 : index
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

func.func @test_work_group_id(%i: i32) -> !sycl_id_1_ {
  // expected-error @+1 {{'sycl.work_group_id' op Expecting an index return value for this cardinality}}
  %0 = sycl.work_group_id(%i) : (i32) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

func.func @test_work_group_id_dim() -> index {
  // expected-error @+1 {{'sycl.work_group_id' op Not expecting an index return value for this cardinality}}
  %0 = sycl.work_group_id() : () -> index
  return %0 : index
}

// -----

func.func @test_num_work_items_dim() -> index {
  %c-1_i32 = arith.constant -1 : i32
  // expected-error @+1 {{'sycl.num_work_items' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.num_work_items(%c-1_i32) : (i32) -> index
  return %0 : index
}

// -----

func.func @test_global_id_dim() -> index {
  %c7_i32 = arith.constant 7 : i32
  // expected-error @+1 {{'sycl.global_id' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.global_id(%c7_i32) : (i32) -> index
  return %0 : index
}

// -----

func.func @test_local_id_dim() -> index {
  %c-1_i32 = arith.constant -1 : i32
  // expected-error @+1 {{'sycl.local_id' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.local_id(%c-1_i32) : (i32) -> index
  return %0 : index
}

// -----

func.func @test_global_offset_dim() -> index {
  %c7_i32 = arith.constant 7 : i32
  // expected-error @+1 {{'sycl.global_offset' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.global_offset(%c7_i32) : (i32) -> index
  return %0 : index
}

// -----

func.func @test_num_work_groups_dim() -> index {
  %c-1_i32 = arith.constant -1 : i32
  // expected-error @+1 {{'sycl.num_work_groups' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.num_work_groups(%c-1_i32) : (i32) -> index
  return %0 : index
}

// -----

func.func @test_work_group_size_dim() -> index {
  %c7_i32 = arith.constant 7 : i32
  // expected-error @+1 {{'sycl.work_group_size' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.work_group_size(%c7_i32) : (i32) -> index
  return %0 : index
}

// -----

func.func @test_work_group_id_dim() -> index {
  %c3_i32 = arith.constant 3 : i32
  // expected-error @+1 {{'sycl.work_group_id' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.work_group_id(%c3_i32) : (i32) -> index
  return %0 : index
}
