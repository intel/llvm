// RUN: sycl-mlir-opt -split-input-file %s -verify-diagnostics

func.func @test_addrspacecast_different_elementtype(
    %arg0: memref<?xi64, #sycl.access.address_space<local>>)
    -> memref<?xi32, #sycl.access.address_space<generic>> {
  // expected-error @+1 {{'sycl.addrspacecast' op operand type 'memref<?xi64, #sycl.access.address_space<local>>' and result type 'memref<?xi32, #sycl.access.address_space<generic>>' are cast incompatible}}
  %0 = sycl.addrspacecast %arg0
      : memref<?xi64, #sycl.access.address_space<local>>
      to memref<?xi32, #sycl.access.address_space<generic>>
  return %0 : memref<?xi32, #sycl.access.address_space<generic>>
}

// -----

func.func @test_addrspacecast_different_shape(
    %arg0: memref<1xi32, #sycl.access.address_space<local>>)
    -> memref<?xi32, #sycl.access.address_space<generic>> {
  // expected-error @+1 {{'sycl.addrspacecast' op operand type 'memref<1xi32, #sycl.access.address_space<local>>' and result type 'memref<?xi32, #sycl.access.address_space<generic>>' are cast incompatible}}
  %0 = sycl.addrspacecast %arg0
      : memref<1xi32, #sycl.access.address_space<local>>
      to memref<?xi32, #sycl.access.address_space<generic>>
  return %0 : memref<?xi32, #sycl.access.address_space<generic>>
}

// -----

func.func @test_addrspacecast_different_layout(
    %arg0: memref<?xi32, affine_map<(d0) -> (d0 + 1)>, #sycl.access.address_space<local>>)
    -> memref<?xi32, #sycl.access.address_space<generic>> {
  // expected-error @+1 {{'sycl.addrspacecast' op operand type 'memref<?xi32, affine_map<(d0) -> (d0 + 1)>, #sycl.access.address_space<local>>' and result type 'memref<?xi32, #sycl.access.address_space<generic>>' are cast incompatible}}
  %0 = sycl.addrspacecast %arg0
      : memref<?xi32, affine_map<(d0) -> (d0 + 1)>, #sycl.access.address_space<local>>
      to memref<?xi32, #sycl.access.address_space<generic>>
  return %0 : memref<?xi32, #sycl.access.address_space<generic>>
}

// -----

func.func @test_addrspacecast_generic_to_generic(
    %arg0: memref<?xi32, #sycl.access.address_space<generic>>)
    -> memref<?xi32, #sycl.access.address_space<generic>> {
  // expected-error @+1 {{'sycl.addrspacecast' op operand type 'memref<?xi32, #sycl.access.address_space<generic>>' and result type 'memref<?xi32, #sycl.access.address_space<generic>>' are cast incompatible}}
  %0 = sycl.addrspacecast %arg0 : memref<?xi32, #sycl.access.address_space<generic>> to memref<?xi32, #sycl.access.address_space<generic>>
  return %0 : memref<?xi32, #sycl.access.address_space<generic>>
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

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

func.func @test_accessor_get_pointer(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi64, 1> {
  // expected-error @+1 {{'sycl.accessor.get_pointer' op Expecting a reference to this accessor's value type}}
  %0 = sycl.accessor.get_pointer(%acc) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi64, 1>
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
  %0 = sycl.accessor.get_range(%acc) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> !sycl_range_2_
  return %0 : !sycl_range_2_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 3>)>)>
!sycl_atomic_i64_glo = !sycl.atomic<[i64, global], (memref<?xi64, 1>)>

func.func @test_accessor_subscript_atomic(
  %acc: memref<?x!sycl_accessor_1_i32_ato_gb>,
  %idx: memref<?x!sycl_id_1_>) -> !sycl_atomic_i64_glo {
  // expected-error @+1 {{'sycl.accessor.subscript' op Expecting a reference to this accessor's value type}}
  %0 = sycl.accessor.subscript %acc[%idx]
      : (memref<?x!sycl_accessor_1_i32_ato_gb>, memref<?x!sycl_id_1_>)
      -> !sycl_atomic_i64_glo
  return %0 : !sycl_atomic_i64_glo
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_2_ = !sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl_item_base_2_)>

func.func @test_get_id_bad_dimensions(%item: memref<?x!sycl_item_2_>) -> i64 {
  // expected-error @below {{'sycl.item.get_id' op operand 0 must have a single dimension to be passed as the single argument to this operation}}
  %0 = sycl.item.get_id(%item) : (memref<?x!sycl_item_2_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

func.func @test_get_id_diff_dim(%item: memref<?x!sycl_item_1_>) -> !sycl_id_2_ {
  // expected-error @below {{'sycl.item.get_id' op base type and return type dimensions mismatch: 1 vs 2}}
  %0 = sycl.item.get_id(%item) : (memref<?x!sycl_item_1_>) -> !sycl_id_2_
  return %0 : !sycl_id_2_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

func.func @test_get_id_bad_type(%item: memref<?x!sycl_item_1_>, %i: i32) -> !sycl_id_1_ {
  // expected-error @below {{'sycl.item.get_id' op must be passed a single argument in order to define an id value}}
  %0 = sycl.item.get_id(%item, %i) : (memref<?x!sycl_item_1_>, i32) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>

func.func @test_get_component_bad_dimensions(%id: memref<?x!sycl_id_3_>) -> i64 {
  // expected-error @below {{'sycl.id.get' op operand 0 must have a single dimension to be passed as the single argument to this operation}}
  %0 = sycl.id.get %id[] : (memref<?x!sycl_id_3_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>

func.func @test_get_component_bad_type(%id: memref<?x!sycl_id_1_>) -> memref<?xi64> {
  // expected-error @below {{'sycl.id.get' op must return a scalar type when a single argument is provided}}
  %0 = sycl.id.get %id[] : (memref<?x!sycl_id_1_>) -> memref<?xi64>
  return %0 : memref<?xi64>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

func.func @test_get_range_bad_scalar_type(%item: memref<?x!sycl_item_1_>) -> i64 {
  // expected-error @below {{'sycl.item.get_range' op expecting range result type. Got 'i64'}}
  %0 = sycl.item.get_range(%item) : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

func.func @test_get_range_diff_dimensions(%item: memref<?x!sycl_item_1_>) -> !sycl_range_2_ {
  // expected-error @below {{'sycl.item.get_range' op base type and return type dimensions mismatch: 1 vs 2}}
  %0 = sycl.item.get_range(%item) : (memref<?x!sycl_item_1_>) -> !sycl_range_2_
  return %0 : !sycl_range_2_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

func.func @test_get_range_bad_ret_type(%item: memref<?x!sycl_item_1_>, %i: i32) -> !sycl_range_1_ {
  // expected-error @below {{'sycl.item.get_range' op expecting an I64 result type. Got '!sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>'}}
  %0 = sycl.item.get_range(%item, %i) : (memref<?x!sycl_item_1_>, i32) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

func.func @test_get_group_bad_scalar_type(%nd: memref<?x!sycl_nd_item_1_>) -> i64 {
  // expected-error @below {{'sycl.nd_item.get_group' op expecting group result type. Got 'i64'}}
  %0 = sycl.nd_item.get_group(%nd) : (memref<?x!sycl_nd_item_1_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

func.func @test_get_group_diff_dimensions(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_group_2_ {
  // expected-error @below {{'sycl.nd_item.get_group' op base type and return type dimensions mismatch: 1 vs 2}}
  %0 = sycl.nd_item.get_group(%nd) : (memref<?x!sycl_nd_item_1_>) -> !sycl_group_2_
  return %0 : !sycl_group_2_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

func.func @test_get_group_bad_ret_type(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> !sycl_group_1_ {
  // expected-error @below {{'sycl.nd_item.get_group' op expecting an I64 result type. Got '!sycl.group<[1], (!sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>, !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>, !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>, !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>)>'}}
  %0 = sycl.nd_item.get_group(%nd, %i) : (memref<?x!sycl_nd_item_1_>, i32) -> !sycl_group_1_
  return %0 : !sycl_group_1_
}
