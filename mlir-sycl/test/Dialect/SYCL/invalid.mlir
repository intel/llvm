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

// -----

// expected-error @below {{Integer SYCL vector element types can only be i1, i8, i16, i32 or i64. Got 'i2'.}}
func.func @test_vec_i2(%v: !sycl.vec<[i2, 2], (vector<2xi2>)>) {
  return
}

// -----

// expected-error @below {{FP SYCL vector element types can only be half, float or double. Got 'f80'.}}
func.func @test_vec_f80(%v: !sycl.vec<[f80, 2], (vector<2xi2>)>) {
  return
}

// -----

!sycl_vec_i8_1_ = !sycl.vec<[i8, 1], (vector<1xi8>)>

// expected-error @below {{SYCL vector types can only hold basic scalar types. Got '!sycl.vec<[i8, 1], (vector<1xi8>)>'.}}
func.func @test_vec_nonscalar(%v: !sycl.vec<[!sycl_vec_i8_1_, 2], (vector<2x1xi8>)>) {
  return
}

// -----

func.func @test_host_constructor() -> !llvm.ptr {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
// expected-error @below {{'sycl.host.constructor' op expecting a sycl type as constructed type. Got 'i32'}}
  sycl.host.constructor(%1) {type = i32} : (!llvm.ptr) -> ()
  return %1 : !llvm.ptr
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>

func.func @test_id_constructor_index_wrong_dims(%arg0: index)
    -> memref<1x!sycl_id_2_> {
// expected-error @below {{'sycl.id.constructor' op expects to be passed the same number of 'index' numbers as the number of dimensions of the input: 1 vs 2}}
  %0 = sycl.id.constructor(%arg0) : (index) -> memref<1x!sycl_id_2_>
  func.return %0 : memref<1x!sycl_id_2_>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>

func.func @test_id_constructor_wrong_copy_dims(%arg: memref<?x!sycl_id_2_>)
    -> memref<1x!sycl_id_1_> {
// expected-error @below {{'sycl.id.constructor' op expects input and output to have the same number of dimensions: 2 vs 1}}
  %0 = sycl.id.constructor(%arg) : (memref<?x!sycl_id_2_>) -> memref<1x!sycl_id_1_>
  func.return %0 : memref<1x!sycl_id_1_>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>

func.func @test_id_constructor_wrong_sign(%arg: i32) -> memref<1x!sycl_id_1_> {
// expected-error @below {{'sycl.id.constructor' op expects a different signature. Check documentation for details}}
  %0 = sycl.id.constructor(%arg) : (i32) -> memref<1x!sycl_id_1_>
  func.return %0 : memref<1x!sycl_id_1_>
}

// -----

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>

func.func @test_range_constructor_bad_signature(%arg: i32) -> memref<1x!sycl_range_1_> {
// expected-error @below {{'sycl.range.constructor' op expects a different signature. Check documentation for details}}
  %0 = sycl.range.constructor(%arg) : (i32) -> memref<1x!sycl_range_1_>
  func.return %0 : memref<1x!sycl_range_1_>
}

// -----

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>

func.func @test_range_constructor_bad_num_dims(%arg: memref<1x!sycl_range_2_>) -> memref<1x!sycl_range_1_> {
// expected-error @below {{'sycl.range.constructor' op expects input and output to have the same number of dimensions: 2 vs 1}}
  %0 = sycl.range.constructor(%arg) : (memref<1x!sycl_range_2_>) -> memref<1x!sycl_range_1_>
  func.return %0 : memref<1x!sycl_range_1_>
}

// -----

!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>

func.func @test_range_constructor_index_wrong_dims(%arg0: index)
    -> memref<1x!sycl_range_2_> {
// expected-error @below {{'sycl.range.constructor' op expects to be passed the same number of 'index' numbers as the number of dimensions of the input: 1 vs 2}}
  %0 = sycl.range.constructor(%arg0) : (index) -> memref<1x!sycl_range_2_>
  func.return %0 : memref<1x!sycl_range_2_>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

func.func @test_nd_constructor_bad_signature(%arg: i32) -> memref<1x!sycl_nd_range_1_> {
// expected-error @below {{'sycl.nd_range.constructor' op expects a different signature. Check documentation for details}}
  %0 = sycl.nd_range.constructor(%arg) : (i32) -> memref<1x!sycl_nd_range_1_>
  func.return %0 : memref<1x!sycl_nd_range_1_>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

func.func @test_nd_range_default() -> memref<1x!sycl_nd_range_1_> {
// expected-error @below {{'sycl.nd_range.constructor' op expects a different signature. Check documentation for details}}
  %nd1 = sycl.nd_range.constructor() : () -> memref<1x!sycl_nd_range_1_>
  func.return %nd1 : memref<1x!sycl_nd_range_1_>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

func.func @test_nd_constructor_bad_local_size(%globalSize: memref<?x!sycl_range_1_>, %localSize: memref<?x!sycl_range_2_>, %offset: memref<?x!sycl_id_1_>) -> memref<1x!sycl_nd_range_1_> {
// expected-error @below {{'sycl.nd_range.constructor' op expects input and output to have the same number of dimensions: 2 vs 1}}
  %0 = sycl.nd_range.constructor(%globalSize, %localSize, %offset) : (memref<?x!sycl_range_1_>, memref<?x!sycl_range_2_>, memref<?x!sycl_id_1_>) -> memref<1x!sycl_nd_range_1_>
  func.return %0 : memref<1x!sycl_nd_range_1_>
}

// -----

func.func @math_op_invalid_type(%arg0 : i32) {
  // expected-error @+1 {{op operand #0 must be 32-bit float or 64-bit float or sycl::half or a sycl::vec of float, double or sycl::half, but got 'i32'}}
  %0 = sycl.math.sin %arg0 : i32
  return
}

// -----

!sycl_vec_i32_4_ = !sycl.vec<[i32, 4], (vector<4xi32>)>

func.func @math_op_invalid_vector_type(%arg0 : !sycl_vec_i32_4_) {
  // expected-error @+1 {{op operand #0 must be 32-bit float or 64-bit float or sycl::half or a sycl::vec of float, double or sycl::half, but got '!sycl.vec<[i32, 4], (vector<4xi32>)>'}}
  %0 = sycl.math.sin %arg0 : !sycl_vec_i32_4_
  return
}

// -----

// COM: Check inexistent symbol.

func.func @f() -> !llvm.ptr {
  // expected-error @below {{'sycl.host.get_kernel' op '@kernels::@k0' does not reference a valid kernel}}
  %0 = sycl.host.get_kernel @kernels::@k0 : !llvm.ptr
  func.return %0 : !llvm.ptr
}

// -----

// COM: Check function is not a gpu.func

func.func @f() -> !llvm.ptr {
  // expected-error @below {{'sycl.host.get_kernel' op '@f0' does not reference a valid kernel}}
  %0 = sycl.host.get_kernel @f0 : !llvm.ptr
  func.return %0 : !llvm.ptr
}

func.func @f0() {
  func.return
}

// -----

// COM: Check function is not a kernel

gpu.module @kernels {
  gpu.func @k0() {
    gpu.return
  }
}

func.func @f() -> !llvm.ptr {
  // expected-error @below {{'sycl.host.get_kernel' op '@kernels::@k0' does not reference a valid kernel}}
  %0 = sycl.host.get_kernel @kernels::@k0 : !llvm.ptr
  func.return %0 : !llvm.ptr
}

// -----

// COM: Check inexistent symbol.

func.func @f(%handler: !llvm.ptr) {
  // expected-error @below {{'sycl.host.handler.set_kernel' op '@kernels::@k0' does not reference a valid kernel}}
  sycl.host.handler.set_kernel %handler -> @kernels::@k0 : !llvm.ptr
  func.return
}

// -----

// COM: Check function is not a gpu.func

func.func @f(%handler: !llvm.ptr) {
  // expected-error @below {{'sycl.host.handler.set_kernel' op '@f0' does not reference a valid kernel}}
  sycl.host.handler.set_kernel %handler -> @f0 : !llvm.ptr
  func.return
}

func.func @f0() {
  func.return
}

// -----

// COM: Check function is not a kernel

func.func @f(%handler: !llvm.ptr) {
  // expected-error @below {{'sycl.host.handler.set_kernel' op '@kernels::@k0' does not reference a valid kernel}}
  sycl.host.handler.set_kernel %handler -> @kernels::@k0 : !llvm.ptr
  func.return
}

gpu.module @kernels {
  gpu.func @k0() {
    gpu.return
  }
}

// -----

func.func @test_wrap(%arg0 : f32) {
  // expected-error @below {{'sycl.mlir.wrap' op operand type 'f32' and result type '!sycl.half<(f16)>' are cast incompatible}}
  %0 = sycl.mlir.wrap %arg0 : f32 to !sycl.half<(f16)>
  return
}

// -----

func.func @test_unwrap(%arg0 : !sycl.half<(f16)>) {
  // expected-error @below {{'sycl.mlir.unwrap' op operand type '!sycl.half<(f16)>' and result type 'i16' are cast incompatible}}
  %0 = sycl.mlir.unwrap %arg0 : !sycl.half<(f16)> to i16
  return
}

// -----

func.func @set_nd_range_unexpected_offset(%handler: !llvm.ptr, %nd_range: !llvm.ptr, %offset: !llvm.ptr) {
  // expected-error @below {{'sycl.host.handler.set_nd_range' op expects no offset argument if the nd_range attribute is set}}
  sycl.host.handler.set_nd_range %handler -> nd_range %nd_range, offset %offset : !llvm.ptr, !llvm.ptr, !llvm.ptr
  func.return
}

// -----

func.func @set_captured_scalar_type_attribute(%lambda: !llvm.ptr, %value: i32) {
  // expected-error @below {{'sycl.host.set_captured' op does not expect a type attribute for a non-pointer value}}
  sycl.host.set_captured %lambda[2] = %value : !llvm.ptr, i32 (i32)
}

// -----

func.func @set_captured_non_sycl_type_attribute(%lambda: !llvm.ptr, %value: !llvm.ptr) {
  // expected-error @below {{'sycl.host.set_captured' op expects the type attribute to reference a SYCL type}}
  sycl.host.set_captured %lambda[2] = %value : !llvm.ptr, !llvm.ptr (i32)
}

// -----

func.func @f(%handler: !llvm.ptr) {
  // expected-error @below {{'sycl.host.schedule_kernel' op '@kernels::@k0' does not reference a valid kernel}}
  sycl.host.schedule_kernel %handler -> @kernels::@k0 : (!llvm.ptr) -> ()
  func.return
}

gpu.module @kernels {
  gpu.func @k0() {
    gpu.return
  }
}

// -----

func.func @schedule_kernel_nd_range_unexpected_attr(%handler: !llvm.ptr) {
  // expected-error @below {{'sycl.host.schedule_kernel' op expects nd_range to be unset when a range is not present}}
  sycl.host.schedule_kernel %handler -> @ekernels::@k0 {nd_range} : (!llvm.ptr) -> ()
  func.return
}

gpu.module @kernels {
  gpu.func @k0() kernel {
    gpu.return
  }
}

// -----

func.func @schedule_kernel_nd_range_unexpected_offset(%handler: !llvm.ptr, %nd_range: !llvm.ptr, %offset: !llvm.ptr) {
  // expected-error @below {{'sycl.host.schedule_kernel' op expects no offset argument if the nd_range attribute is set}}
  sycl.host.schedule_kernel %handler -> @ekernels::@k0[nd_range %nd_range, offset %offset] : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  func.return
}

gpu.module @kernels {
  gpu.func @k0() kernel {
    gpu.return
  }
}

// -----

func.func @schedule_kernel_inconsistent_type_attributes(%handler: !llvm.ptr, %arg0: i32) {
  // expected-error @below {{'sycl.host.schedule_kernel' op has inconsistent SYCL type attributes}}
  "sycl.host.schedule_kernel"(%handler, %arg0) {kernel_name = @kernels::@k0, operand_segment_sizes = array<i32: 1, 0, 0, 1>, sycl_types = [none, none]} : (!llvm.ptr, i32) -> ()
  func.return
}

gpu.module @kernels {
  gpu.func @k0() kernel {
    gpu.return
  }
}

// -----

func.func @schedule_kernel_scalar_type_attribute(%handler: !llvm.ptr, %arg0: i32) {
  // expected-error @below {{'sycl.host.schedule_kernel' op does not expect a type attribute for a non-pointer value}}
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%arg0: i32) : (!llvm.ptr, i32) -> ()
  func.return
}

gpu.module @kernels {
  gpu.func @k0() kernel {
    gpu.return
  }
}

// -----

func.func @schedule_kernel_non_sycl_type_attribute(%handler: !llvm.ptr, %arg0: !llvm.ptr) {
  // expected-error @below {{'sycl.host.schedule_kernel' op expects the type attribute to reference a SYCL type}}
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%arg0: i32) : (!llvm.ptr, !llvm.ptr) -> ()
  func.return
}

gpu.module @kernels {
  gpu.func @k0() kernel {
    gpu.return
  }
}

// -----

func.func @f(%event: !llvm.ptr, %queue: !llvm.ptr) {
  // expected-error @below {{'sycl.host.submit' op '@f0' does not reference a valid CGF function}}
  sycl.host.submit %queue(@f0) -> %event : !llvm.ptr, !llvm.ptr
  func.return
}

func.func @f0() {
  func.return
}

// -----

func.func @f(%event: !llvm.ptr, %queue: !llvm.ptr) {
  // expected-error @below {{'sycl.host.submit' op expects CGF function to have internal linkage}}
  // expected-note @below {{got: 'external'}}
  sycl.host.submit %queue(@f0) -> %event : !llvm.ptr, !llvm.ptr
  func.return
}

llvm.func @f0(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  llvm.return
}

// -----

func.func @f(%event: !llvm.ptr, %queue: !llvm.ptr) {
  // expected-error @below {{'sycl.host.submit' op expects CGF function to not have variadic arguments}}
  sycl.host.submit %queue(@f0) -> %event : !llvm.ptr, !llvm.ptr
  func.return
}

llvm.func internal @f0(%arg0: !llvm.ptr, ...) {
  llvm.return
}

// -----

func.func @f(%event: !llvm.ptr, %queue: !llvm.ptr) {
  // expected-error @below {{'sycl.host.submit' op incorrect number of operands for CGF}}
  sycl.host.submit %queue(@f0) -> %event : !llvm.ptr, !llvm.ptr
  func.return
}

llvm.func internal @f0(%arg1: i64) {
  llvm.return
}

// -----

func.func @f(%event: !llvm.ptr, %queue: !llvm.ptr) {
  // expected-error @below {{'sycl.host.submit' op expecting CGF's operand type '!llvm.ptr', but got 'i64' for operand number 0}}
  sycl.host.submit %queue(@f0) -> %event : !llvm.ptr, !llvm.ptr
  func.return
}

llvm.func internal @f0(%arg0: i64, %arg1: !llvm.ptr) {
  llvm.return
}

// -----

func.func @f(%event: !llvm.ptr, %queue: !llvm.ptr) {
  // expected-error @below {{'sycl.host.submit' op expecting CGF's result type '!llvm.void', but got 'i64'}}
  sycl.host.submit %queue(@f0) -> %event : !llvm.ptr, !llvm.ptr
  func.return
}

llvm.func internal @f0(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i64 {
  %0 = llvm.mlir.constant(0 : i64) : i64
  llvm.return %0 : i64
}
