// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK-DAG: !sycl_id_1_ = !sycl.id<1>
// CHECK-DAG: !sycl_id_2_ = !sycl.id<2>
// CHECK-DAG: !sycl_range_1_ = !sycl.range<1>
// CHECK-DAG: !sycl_range_2_ = !sycl.range<2>
// CHECK-DAG: !sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK-DAG: !sycl_accessor_2_i32_read_write_global_buffer = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK-DAG: !sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
// CHECK-DAG: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
// CHECK-DAG: !sycl_group_1_ = !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>
// CHECK-DAG: !sycl_group_2_ = !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>
// CHECK-DAG: !sycl_item_1_1_ = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>
// CHECK-DAG: !sycl_item_2_0_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>
// CHECK-DAG: !sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>, !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl.range<1>, !sycl.id<1>)>)>, !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>)>
// CHECK-DAG: !sycl_nd_item_2_ = !sycl.nd_item<[2], (!sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>, !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>, !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>)>
// CHECK-DAG: !sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>
// CHECK-DAG: !sycl_nd_range_2_ = !sycl.nd_range<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>
// CHECK-DAG: !sycl_atomic_f32_3_ = !sycl.atomic<[f32,3], (memref<?xf32, 3>)>
// CHECK-DAG: !sycl_atomic_i32_1_ = !sycl.atomic<[i32,1], (memref<?xi32, 1>)>

// CHECK-LABEL: func.func @_Z4id_1N4sycl3_V12idILi1EEE(
// CHECK:          %arg0: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC:llvm.cconv = #llvm.cconv<spir_funccc>]], [[LINKEXT:llvm.linkage = #llvm.linkage<external>]],
// CHECK-SAME: [[PASSTHROUGH:passthrough = \["convergent", "mustprogress", "norecurse", "nounwind", \["frame-pointer", "all"\], \["no-trapping-math", "true"\], \["stack-protector-buffer-size", "8"\], \["sycl-module-id", ".*/polygeist/tools/cgeist/Test/Verification/sycl/types.cpp"\]\]]]} {
SYCL_EXTERNAL void id_1(sycl::id<1> id) {}

// CHECK-LABEL: func.func @_Z4id_2N4sycl3_V12idILi2EEE(
// CHECK:          %arg0: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void id_2(sycl::id<2> id) {}

// CHECK-LABEL: func.func @_Z5acc_1N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// CHECK:          %arg0: memref<?x!sycl_accessor_1_i32_read_write_global_buffer> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_1_i32_read_write_global_buffer, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void acc_1(sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write>) {}

// CHECK-LABEL: func.func @_Z5acc_2N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// CHECK:          %arg0: memref<?x!sycl_accessor_2_i32_read_write_global_buffer> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_2_i32_read_write_global_buffer, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void acc_2(sycl::accessor<sycl::cl_int, 2, sycl::access::mode::read_write>) {}

// CHECK-LABEL: func.func @_Z5acc_3N4sycl3_V18accessorIfLi3ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(
// CHECK:          %arg0: memref<?x!sycl_accessor_3_f32_read_write_global_buffer> {llvm.align = 8 : i64, llvm.byval = !sycl_accessor_3_f32_read_write_global_buffer, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void acc_3(sycl::accessor<sycl::cl_float, 3, sycl::access::mode::read_write>) {}

// CHECK-LABEL: func.func @_Z7range_1N4sycl3_V15rangeILi1EEE(
// CHECK:          %arg0: memref<?x!sycl_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void range_1(sycl::range<1> range) {}

// CHECK-LABEL: func.func @_Z7range_2N4sycl3_V15rangeILi2EEE(
// CHECK:          %arg0: memref<?x!sycl_range_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_2_, llvm.noundef})  
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
// CHECK:          %arg0: memref<?x!sycl_item_1_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_item_1_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void item_1_true(sycl::item<1, true> item) {}

// CHECK-LABEL: func.func @_Z12item_2_falseN4sycl3_V14itemILi2ELb0EEE(
// CHECK:          %arg0: memref<?x!sycl_item_2_0_> {llvm.align = 8 : i64, llvm.byval = !sycl_item_2_0_, llvm.noundef})
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

// CHECK-LABEL: func.func @_Z8atomic_1N4sycl3_V16atomicIiLNS0_6access13address_spaceE1EEE(
// CHECK:          %arg0: memref<?x!sycl_atomic_i32_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_atomic_i32_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void atomic_1(sycl::atomic<int> atomic_int) {}

// CHECK-LABEL: func.func @_Z8atomic_2N4sycl3_V16atomicIfLNS0_6access13address_spaceE3EEE(
// CHECK:          %arg0: memref<?x!sycl_atomic_f32_3_> {llvm.align = 8 : i64, llvm.byval = !sycl_atomic_f32_3_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXT]], [[PASSTHROUGH]]
SYCL_EXTERNAL void atomic_2(sycl::atomic<float, sycl::access::address_space::local_space> atomic_float) {}


