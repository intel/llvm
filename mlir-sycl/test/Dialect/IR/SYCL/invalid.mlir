// RUN: sycl-mlir-opt -split-input-file %s -verify-diagnostics

func.func @test_addrspacecast_different_elementtype(%arg0: memref<?xi64>) -> memref<?xi32, 4> {
  // expected-error @+1 {{'sycl.addrspacecast' op operand type 'memref<?xi64>' and result type 'memref<?xi32, 4>' are cast incompatible}}
  %0 = sycl.addrspacecast %arg0 : memref<?xi64> to memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// -----

func.func @test_addrspacecast_different_shape(%arg0: memref<1xi32>) -> memref<?xi32, 4> {
  // expected-error @+1 {{'sycl.addrspacecast' op operand type 'memref<1xi32>' and result type 'memref<?xi32, 4>' are cast incompatible}}
  %0 = sycl.addrspacecast %arg0 : memref<1xi32> to memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// -----

func.func @test_addrspacecast_different_layout(%arg0: memref<?xi32, affine_map<(d0) -> (d0 + 1)>>) -> memref<?xi32, 4> {
  // expected-error @+1 {{'sycl.addrspacecast' op operand type 'memref<?xi32, affine_map<(d0) -> (d0 + 1)>>' and result type 'memref<?xi32, 4>' are cast incompatible}}
  %0 = sycl.addrspacecast %arg0 : memref<?xi32, affine_map<(d0) -> (d0 + 1)>> to memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// -----

func.func @test_addrspacecast_generic_to_generic(%arg0: memref<?xi32, 4>) -> memref<?xi32, 4> {
  // expected-error @+1 {{'sycl.addrspacecast' op operand type 'memref<?xi32, 4>' and result type 'memref<?xi32, 4>' are cast incompatible}}
  %0 = sycl.addrspacecast %arg0 : memref<?xi32, 4> to memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>

func.func @test_cast_not_parents(%arg: memref<1x!sycl_id_1_>) -> memref<1x!sycl.accessor_common> {
  // expected-error @+1 {{'sycl.cast' op operand type 'memref<1x!sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>>' and result type 'memref<1x!sycl.accessor_common>' are cast incompatible}}
  %0 = sycl.cast %arg : memref<1x!sycl_id_1_> to memref<1x!sycl.accessor_common>
  return %0 : memref<1x!sycl.accessor_common>
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>

func.func @test_cast_bad_shape(%arg: memref<1x!sycl_id_1_>) -> memref<2x!sycl_array_1_> {
  // expected-error @+1 {{'sycl.cast' op operand type 'memref<1x!sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>>' and result type 'memref<2x!sycl.array<[1], (memref<1xi64, 4>)>>' are cast incompatible}}
  %0 = sycl.cast %arg : memref<1x!sycl_id_1_> to memref<2x!sycl_array_1_>
  return %0 : memref<2x!sycl_array_1_>
}

// -----

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

func.func @test_non_memref_arg_constructor(%range: !sycl_range_1_) {
  // expected-error @+1 {{'sycl.constructor' op operand #0 must be memref}}
  sycl.constructor @range(%range) {MangledFunctionName = @rangev} : (!sycl_range_1_)
}

// -----

func.func @test_non_sycl_arg_constructor(%i: memref<1xi32>) {
  // expected-error @+1 {{'sycl.constructor' op operand #0 must be memref}}  
  sycl.constructor @range(%i) {MangledFunctionName = @rangev} : (memref<1xi32>)
}

// -----

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

func.func @test_num_work_items(%i: i32) -> !sycl_range_1_ {
  // expected-error @+1 {{'sycl.num_work_items' op Expecting an index return value for this cardinality}}
  %0 = sycl.num_work_items %i : !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

func.func @test_num_work_items_dim() -> index {
  // expected-error @+1 {{'sycl.num_work_items' op Not expecting an index return value for this cardinality}}
  %0 = sycl.num_work_items : index
  return %0 : index
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>

func.func @test_global_id(%i: i32) -> !sycl_id_2_ {
  // expected-error @+1 {{'sycl.global_id' op Expecting an index return value for this cardinality}}
  %0 = sycl.global_id %i : !sycl_id_2_
  return %0 : !sycl_id_2_
}

// -----

func.func @test_global_id_dim() -> index {
  // expected-error @+1 {{'sycl.global_id' op Not expecting an index return value for this cardinality}}
  %0 = sycl.global_id : index
  return %0 : index
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

func.func @test_local_id(%i: i32) -> !sycl_id_3_ {
  // expected-error @+1 {{'sycl.local_id' op Expecting an index return value for this cardinality}}
  %0 = sycl.local_id %i : !sycl_id_3_
  return %0 : !sycl_id_3_
}

// -----

func.func @test_local_id_dim() -> index {
  // expected-error @+1 {{'sycl.local_id' op Not expecting an index return value for this cardinality}}
  %0 = sycl.local_id : index
  return %0 : index
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

func.func @test_global_offset(%i: i32) -> !sycl_id_1_ {
  // expected-error @+1 {{'sycl.global_offset' op Expecting an index return value for this cardinality}}
  %0 = sycl.global_offset %i : !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

func.func @test_global_offset_dim() -> index {
  // expected-error @+1 {{'sycl.global_offset' op Not expecting an index return value for this cardinality}}
  %0 = sycl.global_offset : index
  return %0 : index
}

// -----

!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>

func.func @test_num_work_groups(%i: i32) -> !sycl_range_2_ {
  // expected-error @+1 {{'sycl.num_work_groups' op Expecting an index return value for this cardinality}}
  %0 = sycl.num_work_groups %i : !sycl_range_2_
  return %0 : !sycl_range_2_
}

// -----

func.func @test_num_work_groups_dim() -> index {
  // expected-error @+1 {{'sycl.num_work_groups' op Not expecting an index return value for this cardinality}}
  %0 = sycl.num_work_groups : index
  return %0 : index
}

// -----

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

func.func @test_work_group_size(%i: i32) -> !sycl_range_3_ {
  // expected-error @+1 {{'sycl.work_group_size' op Expecting an index return value for this cardinality}}
  %0 = sycl.work_group_size %i : !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

func.func @test_work_group_size_dim() -> index {
  // expected-error @+1 {{'sycl.work_group_size' op Not expecting an index return value for this cardinality}}
  %0 = sycl.work_group_size : index
  return %0 : index
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

func.func @test_work_group_id(%i: i32) -> !sycl_id_1_ {
  // expected-error @+1 {{'sycl.work_group_id' op Expecting an index return value for this cardinality}}
  %0 = sycl.work_group_id %i : !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

func.func @test_work_group_id_dim() -> index {
  // expected-error @+1 {{'sycl.work_group_id' op Not expecting an index return value for this cardinality}}
  %0 = sycl.work_group_id : index
  return %0 : index
}

// -----

func.func @test_num_work_items_dim() -> index {
  %c-1_i32 = arith.constant -1 : i32
  // expected-error @+1 {{'sycl.num_work_items' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.num_work_items %c-1_i32 : index
  return %0 : index
}

// -----

func.func @test_global_id_dim() -> index {
  %c7_i32 = arith.constant 7 : i32
  // expected-error @+1 {{'sycl.global_id' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.global_id %c7_i32 : index
  return %0 : index
}

// -----

func.func @test_local_id_dim() -> index {
  %c-1_i32 = arith.constant -1 : i32
  // expected-error @+1 {{'sycl.local_id' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.local_id %c-1_i32 : index
  return %0 : index
}

// -----

func.func @test_global_offset_dim() -> index {
  %c7_i32 = arith.constant 7 : i32
  // expected-error @+1 {{'sycl.global_offset' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.global_offset %c7_i32 : index
  return %0 : index
}

// -----

func.func @test_num_work_groups_dim() -> index {
  %c-1_i32 = arith.constant -1 : i32
  // expected-error @+1 {{'sycl.num_work_groups' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.num_work_groups %c-1_i32 : index
  return %0 : index
}

// -----

func.func @test_work_group_size_dim() -> index {
  %c7_i32 = arith.constant 7 : i32
  // expected-error @+1 {{'sycl.work_group_size' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.work_group_size %c7_i32 : index
  return %0 : index
}

// -----

func.func @test_work_group_id_dim() -> index {
  %c3_i32 = arith.constant 3 : i32
  // expected-error @+1 {{'sycl.work_group_id' op The SYCL index space can only be 1, 2, or 3 dimensional}}
  %0 = sycl.work_group_id %c3_i32 : index
  return %0 : index
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

func.func @test_accessor_get_pointer(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi64, 1> {
  // expected-error @+1 {{'sycl.accessor.get_pointer' op Expecting a reference to this accessor's value type}}
  %0 = sycl.accessor.get_pointer(%acc) { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>], FunctionName = @"get_pointer", MangledFunctionName = @"get_pointer", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi64, 1>
  return %0 : memref<?xi64, 1>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>

func.func @test_accessor_get_range(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> !sycl_range_2_ {
  // expected-error @+1 {{'sycl.accessor.get_range' op Both the result and the accessor must have the same number of dimensions, but the accessor has 1 dimension(s) and the result has 2 dimension(s)}}
  %0 = sycl.accessor.get_range(%acc) { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>], FunctionName = @"get_range", MangledFunctionName = @"get_range", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> !sycl_range_2_
  return %0 : !sycl_range_2_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 3>)>)>
!sycl_atomic_i64_glo_ = !sycl.atomic<[i64, global], (memref<?xi64, 1>)>

func.func @test_accessor_subscript_atomic(
  %acc: memref<?x!sycl_accessor_1_i32_ato_gb>, 
  %idx: memref<?x!sycl_id_1_>) -> !sycl_atomic_i64_glo_ {
  // expected-error @+1 {{'sycl.accessor.subscript' op Expecting a reference to this accessor's value type}}
  %0 = sycl.accessor.subscript %acc[%idx] { 
        ArgumentTypes = [memref<?x!sycl_accessor_1_i32_ato_gb>, 
                          memref<?x!sycl_id_1_>], 
        FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_i32_ato_gb>, 
                                memref<?x!sycl_id_1_>) -> !sycl_atomic_i64_glo_
  return %0 : !sycl_atomic_i64_glo_
}
