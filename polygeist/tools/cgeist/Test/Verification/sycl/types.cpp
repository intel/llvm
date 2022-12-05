// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK-DAG: !sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK-DAG: !sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK-DAG: !sycl_accessor_3_f32_rw_gb = !sycl.accessor<[3, f32, read_write, global_buffer], (!sycl.accessor_impl_device<[3], (!sycl.id<3>, !sycl.range<3>, !sycl.range<3>)>, !llvm.struct<(ptr<f32, 1>)>)>
// CHECK-DAG: !sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
// CHECK-DAG: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
// CHECK-DAG: !sycl_group_1_ = !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>
// CHECK-DAG: !sycl_group_2_ = !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>
// CHECK-DAG: ![[ITEM1:.*]] = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>
// CHECK-DAG: ![[ITEM2:.*]] = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>
// CHECK-DAG: !sycl_nd_item_1_ = !sycl.nd_item<[1], (![[ITEM1]], !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl.range<1>, !sycl.id<1>)>)>, !sycl_group_1_)>
// CHECK-DAG: !sycl_nd_item_2_ = !sycl.nd_item<[2], (!sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>, ![[ITEM2]], !sycl_group_2_)>
// CHECK-DAG: !sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>
// CHECK-DAG: !sycl_nd_range_2_ = !sycl.nd_range<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>
// CHECK-DAG: !sycl_get_scalar_op_i32_ = !sycl.get_scalar_op<[i32], (i32)>
// CHECK-DAG: !sycl_tuple_value_holder_i32_ = !sycl.tuple_value_holder<[i32], (i32)>
// CHECK-DAG: [[TUPLE_COPY_ASSIGNABLE_VALUE_HOLDER_TRUE:!sycl_tuple_copy_assignable_value_holder_i32_.*]] = !sycl.tuple_copy_assignable_value_holder<[i32, true], (!sycl_tuple_value_holder_i32_)>
// CHECK-DAG: [[TUPLE_COPY_ASSIGNABLE_VALUE_HOLDER_FALSE:!sycl_tuple_copy_assignable_value_holder_i32_.*]] = !sycl.tuple_copy_assignable_value_holder<[i32, false], (!sycl_tuple_value_holder_i32_)>
// CHECK-DAG: !sycl_vec_f32_8_ = !sycl.vec<[f32, 8], (vector<8xf32>)>
// CHECK-DAG: !sycl_vec_i32_4_ = !sycl.vec<[i32, 4], (vector<4xi32>)>
// CHECK-DAG: !sycl_atomic_f32_3_ = !sycl.atomic<[f32,3], (memref<?xf32, 3>)>
// CHECK-DAG: !sycl_atomic_i32_1_ = !sycl.atomic<[i32,1], (memref<?xi32, 1>)>
// CHECK-DAG: !sycl_assert_happened_ = !sycl.assert_happened<(i32, !llvm.array<257 x i8>, !llvm.array<257 x i8>, !llvm.array<129 x i8>, i32, i64, i64, i64, i64, i64, i64)>
// CHECK-DAG: !sycl_bfloat16_ = !sycl.bfloat16<(i16)>
// CHECK-DAG: !sycl_multi_ptr_i32_1_ = !sycl.multi_ptr<[i32, 1, 1], (memref<?xi32, 1>)>

// CHECK-LABEL: func.func @_Z4id_1N4sycl3_V12idILi1EEE(
// CHECK:          %arg0: memref<?x!sycl.id<1>> {llvm.align = 8 : i64, llvm.byval = !sycl.id<1>, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC:llvm.cconv = #llvm.cconv<spir_funccc>]], [[LINKEXT:llvm.linkage = #llvm.linkage<external>]],
// CHECK-SAME: [[PASSTHROUGH:passthrough = \["convergent", "mustprogress", "norecurse", "nounwind", \["frame-pointer", "all"\], \["no-trapping-math", "true"\], \["stack-protector-buffer-size", "8"\], \["sycl-module-id", ".*/polygeist/tools/cgeist/Test/Verification/sycl/types.cpp"\]\]]]} {
SYCL_EXTERNAL void id_1(sycl::id<1> id) {}

// CHECK-LABEL: func.func @_Z4id_2N4sycl3_V12idILi2EEE(
// CHECK:          %arg0: memref<?x!sycl.id<2>> {llvm.align = 8 : i64, llvm.byval = !sycl.id<2>, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void id_2(sycl::id<2> id) {}

// CHECK-LABEL: func.func @_Z5acc_1N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// CHECK:          %arg0: memref<?x!sycl_accessor_1_i32_rw_gb> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_1_i32_rw_gb, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void acc_1(sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write>) {}

// CHECK-LABEL: func.func @_Z5acc_2N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// CHECK:          %arg0: memref<?x!sycl_accessor_2_i32_rw_gb> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_2_i32_rw_gb, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void acc_2(sycl::accessor<sycl::cl_int, 2, sycl::access::mode::read_write>) {}

// CHECK-LABEL: func.func @_Z5acc_3N4sycl3_V18accessorIfLi3ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// CHECK:          %arg0: memref<?x!sycl_accessor_3_f32_rw_gb> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_3_f32_rw_gb, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void acc_3(sycl::accessor<sycl::cl_float, 3, sycl::access::mode::read_write>) {}

// CHECK-LABEL: func.func @_Z7range_1N4sycl3_V15rangeILi1EEE(
// CHECK:          %arg0: memref<?x!sycl.range<1>> {llvm.align = 8 : i64, llvm.byval = !sycl.range<1>, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void range_1(sycl::range<1> range) {}

// CHECK-LABEL: func.func @_Z7range_2N4sycl3_V15rangeILi2EEE(
// CHECK:          %arg0: memref<?x!sycl.range<2>> {llvm.align = 8 : i64, llvm.byval = !sycl.range<2>, llvm.noundef})  
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void range_2(sycl::range<2> range) {}

// CHECK-LABEL: func.func @_Z10nd_range_1N4sycl3_V18nd_rangeILi1EEE(
// CHECK:          %arg0: memref<?x!sycl_nd_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_range_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void nd_range_1(sycl::nd_range<1> nd_range) {}

// CHECK-LABEL: func @_Z10nd_range_2N4sycl3_V18nd_rangeILi2EEE(
// CHECK:          %arg0: memref<?x!sycl_nd_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_range_2_, llvm.noundef})  
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void nd_range_2(sycl::nd_range<2> nd_range) {}

// CHECK-LABEL: func.func @_Z5arr_1N4sycl3_V16detail5arrayILi1EEE(
// CHECK:          %arg0: memref<?x!sycl_array_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_array_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void arr_1(sycl::detail::array<1> arr) {}

// CHECK-LABEL: func.func @_Z5arr_2N4sycl3_V16detail5arrayILi2EEE(
// CHECK:          %arg0: memref<?x!sycl_array_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_array_2_, llvm.noundef})  
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void arr_2(sycl::detail::array<2> arr) {}

// CHECK-LABEL: func.func @_Z11item_1_trueN4sycl3_V14itemILi1ELb1EEE(
// CHECK:          %arg0: memref<?x![[ITEM1]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM1]], llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void item_1_true(sycl::item<1, true> item) {}

// CHECK-LABEL: func.func @_Z12item_2_falseN4sycl3_V14itemILi2ELb0EEE(
// CHECK:          %arg0: memref<?x![[ITEM2]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM2]], llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void item_2_false(sycl::item<2, false> item) {}

// CHECK-LABEL: func.func @_Z9nd_item_1N4sycl3_V17nd_itemILi1EEE(
// CHECK:          %arg0: memref<?x!sycl_nd_item_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void nd_item_1(sycl::nd_item<1> nd_item) {}

// CHECK-LABEL: func.func @_Z9nd_item_2N4sycl3_V17nd_itemILi2EEE(
// CHECK:          %arg0: memref<?x!sycl_nd_item_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_nd_item_2_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void nd_item_2(sycl::nd_item<2> nd_item) {}

// CHECK-LABEL: func.func @_Z7group_1N4sycl3_V15groupILi1EEE(
// CHECK:          %arg0: memref<?x!sycl_group_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void group_1(sycl::group<1> group) {}

// CHECK-LABEL: func.func @_Z7group_2N4sycl3_V15groupILi2EEE(
// CHECK:          %arg0: memref<?x!sycl_group_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_group_2_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void group_2(sycl::group<2> group) {}

// CHECK-LABEL: func.func @_Z6get_opN4sycl3_V16detail5GetOpIiEE(
// CHECK:          %arg0: memref<?x!sycl.get_op<i32>> {llvm.align = 1 : i64, llvm.byval = !sycl.get_op<i32>, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void get_op(sycl::detail::GetOp<int> get_op) {}

// CHECK-LABEL: func.func @_Z13get_scalar_opN4sycl3_V16detail11GetScalarOpIiEE(
// CHECK:          %arg0: memref<?x!sycl_get_scalar_op_i32_> {llvm.align = 4 : i64, llvm.byval = !sycl_get_scalar_op_i32_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void get_scalar_op(sycl::detail::GetScalarOp<int> get_scalar_op) {}

// CHECK-LABEL: func.func @_Z18tuple_value_holderN4sycl3_V16detail16TupleValueHolderIiEE(
// CHECK:          %arg0: memref<?x!sycl_tuple_value_holder_i32_> {llvm.align = 4 : i64, llvm.byval = !sycl_tuple_value_holder_i32_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void tuple_value_holder(sycl::detail::TupleValueHolder<int> get_tuple_value_holder) {}

// CHECK-LABEL: func.func @_Z36tuple_copy_assignable_value_holder_1N4sycl3_V16detail30TupleCopyAssignableValueHolderIiLb1EEE(
// CHECK:          %arg0: memref<?x[[TUPLE_COPY_ASSIGNABLE_VALUE_HOLDER_TRUE]]> {llvm.align = 4 : i64, llvm.byval = [[TUPLE_COPY_ASSIGNABLE_VALUE_HOLDER_TRUE]], llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void tuple_copy_assignable_value_holder_1(sycl::detail::TupleCopyAssignableValueHolder<int, true> tuple_copy_assignable_value_holder_true) {}

// CHECK-LABEL: func.func @_Z36tuple_copy_assignable_value_holder_2N4sycl3_V16detail30TupleCopyAssignableValueHolderIiLb0EEE(
// CHECK:          %arg0: memref<?x[[TUPLE_COPY_ASSIGNABLE_VALUE_HOLDER_FALSE]]> {llvm.align = 4 : i64, llvm.byval = [[TUPLE_COPY_ASSIGNABLE_VALUE_HOLDER_FALSE]], llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void tuple_copy_assignable_value_holder_2(sycl::detail::TupleCopyAssignableValueHolder<int, false> tuple_copy_assignable_value_holder_false) {}

// CHECK-LABEL: func.func @_Z5vec_3N4sycl3_V13vecIiLi4EEE(
// CHECK-SAME:    %arg0: memref<?x!sycl_vec_i32_4_> {llvm.align = 16 : i64, llvm.byval = !sycl_vec_i32_4_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void vec_3(sycl::vec<sycl::cl_int, 4> vec) {}

// CHECK-LABEL: func.func @_Z5vec_4N4sycl3_V13vecIfLi8EEE(
// CHECK-SAME:    %arg0: memref<?x!sycl_vec_f32_8_> {llvm.align = 32 : i64, llvm.byval = !sycl_vec_f32_8_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void vec_4(sycl::float8 vec) {}

// CHECK-LABEL: func.func @_Z8atomic_1N4sycl3_V16atomicIiLNS0_6access13address_spaceE1EEE(
// CHECK:          %arg0: memref<?x!sycl_atomic_i32_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_atomic_i32_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void atomic_1(sycl::atomic<int> atomic_int) {}

// CHECK-LABEL: func.func @_Z8atomic_2N4sycl3_V16atomicIfLNS0_6access13address_spaceE3EEE(
// CHECK:          %arg0: memref<?x!sycl_atomic_f32_3_> {llvm.align = 8 : i64, llvm.byval = !sycl_atomic_f32_3_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void atomic_2(sycl::atomic<float, sycl::access::address_space::local_space> atomic_float) {}

// %"struct.sycl::_V1::detail::AssertHappened" = type { i32, [257 x i8], [257 x i8], [129 x i8], i32, i64, i64, i64, i64, i64, i64 }
// CHECK-LABEL: func.func @_Z19get_assert_happenedN4sycl3_V16detail14AssertHappenedE(
// CHECK:          %arg0: memref<?x!sycl_assert_happened_> {llvm.align = 8 : i64, llvm.byval = !sycl_assert_happened_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void get_assert_happened(sycl::detail::AssertHappened get_assert_happened) {}

// CHECK-LABEL: func.func @_Z12get_bfloat16N4sycl3_V13ext6oneapi8bfloat16E(
// CHECK:          %arg0: memref<?x!sycl_bfloat16_> {llvm.align = 2 : i64, llvm.byval = !sycl_bfloat16_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void get_bfloat16(sycl::ext::oneapi::bfloat16 get_bfloat16) {}

// CHECL-LABEL: func.func @_Z13get_sub_groupN4sycl3_V13ext6oneapi9sub_groupE(
// CHECK:          %arg0: memref<?x!sycl.sub_group> {llvm.align = 1 : i64, llvm.byval = !sycl.sub_group, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void get_sub_group(sycl::ext::oneapi::sub_group get_sub_group) {}

// CHECK-LABEL: func.func @_Z9multi_ptrN4sycl3_V19multi_ptrIiLNS0_6access13address_spaceE1ELNS2_9decoratedE1EEE(
// CHECK:          %arg0: memref<?x!sycl_multi_ptr_i32_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_multi_ptr_i32_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void multi_ptr(sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::yes> multi_ptr_int) {}
