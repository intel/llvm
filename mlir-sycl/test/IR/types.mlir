// Copyright (C) Codeplay Software Limited

//===--- types.mlir -------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verify the type aliase printed output can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

// Verify the generic printed output can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

////////////////////////////////////////////////////////////////////////////////
// ID
////////////////////////////////////////////////////////////////////////////////

!sycl_id_1_ = type !sycl.id<1>
!sycl_id_2_ = type !sycl.id<2>

// CHECK: func @_Z4id_1N2cl4sycl2idILi1EEE(%arg0: !sycl.id<1>)
func @_Z4id_1N2cl4sycl2idILi1EEE(%arg0: !sycl_id_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z4id_2N2cl4sycl2idILi2EEE(%arg0: !sycl.id<2>)
func @_Z4id_2N2cl4sycl2idILi2EEE(%arg0: !sycl_id_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

////////////////////////////////////////////////////////////////////////////////
// ACCESSOR
////////////////////////////////////////////////////////////////////////////////

!sycl_accessor_1_i32_write_global_buffer = type !sycl.accessor<[1, i32, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_2_i32_read_global_buffer = type !sycl.accessor<[2, i32, read, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK: func @_Z5acc_1N2cl4sycl8accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl.accessor<[1, i32, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>)
func @_Z5acc_1N2cl4sycl8accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl_accessor_1_i32_write_global_buffer) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z5acc_2N2cl4sycl8accessorIiLi2ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl.accessor<[2, i32, read, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>, !llvm.struct<(ptr<i32, 1>)>)>)
func @_Z5acc_2N2cl4sycl8accessorIiLi2ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl_accessor_2_i32_read_global_buffer) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

////////////////////////////////////////////////////////////////////////////////
// RANGE
////////////////////////////////////////////////////////////////////////////////

!sycl_range_1_ = type !sycl.range<1>
!sycl_range_2_ = type !sycl.range<2>

// CHECK: func @_Z7range_1N2cl4sycl5rangeILi1EEE(%arg0: !sycl.range<1>)
func @_Z7range_1N2cl4sycl5rangeILi1EEE(%arg0: !sycl_range_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z7range_2N2cl4sycl5rangeILi2EEE(%arg0: !sycl.range<2>)
func @_Z7range_2N2cl4sycl5rangeILi2EEE(%arg0: !sycl_range_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

////////////////////////////////////////////////////////////////////////////////
// ARRAY
////////////////////////////////////////////////////////////////////////////////

!sycl_array_1_ = type !sycl.array<[1], (memref<1xi64>)>
!sycl_array_2_ = type !sycl.array<[2], (memref<2xi64>)>

// CHECK: func @_Z5arr_1N2cl4sycl6detail5arrayILi1EEE(%arg0: !sycl.array<[1], (memref<1xi64>)>)
func @_Z5arr_1N2cl4sycl6detail5arrayILi1EEE(%arg0: !sycl_array_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z5arr_2N2cl4sycl6detail5arrayILi2EEE(%arg0: !sycl.array<[2], (memref<2xi64>)>)
func @_Z5arr_2N2cl4sycl6detail5arrayILi2EEE(%arg0: !sycl_array_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

////////////////////////////////////////////////////////////////////////////////
// ITEM
////////////////////////////////////////////////////////////////////////////////

!sycl_item_1_1_ = type !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>
!sycl_item_2_0_ = type !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>

// CHECK: func @_Z11item_1_trueN2cl4sycl4itemILi1ELb1EEE(%arg0: !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>)
func @_Z11item_1_trueN2cl4sycl4itemILi1ELb1EEE(%arg0: !sycl_item_1_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z12item_2_falseN2cl4sycl4itemILi2ELb0EEE(%arg0: !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>)
func @_Z12item_2_falseN2cl4sycl4itemILi2ELb0EEE(%arg0: !sycl_item_2_0_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

////////////////////////////////////////////////////////////////////////////////
// ND_ITEM 
////////////////////////////////////////////////////////////////////////////////

!sycl_nd_item_1_ = type !sycl.nd_item<[1], (!sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>, !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl.range<1>, !sycl.id<1>)>)>, !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>)>
!sycl_nd_item_2_ = type !sycl.nd_item<[2], (!sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>, !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>, !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>)>

// CHECK: func @_Z9nd_item_1N2cl4sycl7nd_itemILi1EEE(%arg0: !sycl.nd_item<[1], (!sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>, !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl.range<1>, !sycl.id<1>)>)>, !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>)>)
func @_Z9nd_item_1N2cl4sycl7nd_itemILi1EEE(%arg0: !sycl_nd_item_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z9nd_item_2N2cl4sycl7nd_itemILi2EEE(%arg0: !sycl.nd_item<[2], (!sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>, !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl.range<2>, !sycl.id<2>)>)>, !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>)>)
func @_Z9nd_item_2N2cl4sycl7nd_itemILi2EEE(%arg0: !sycl_nd_item_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

////////////////////////////////////////////////////////////////////////////////
// GROUP
////////////////////////////////////////////////////////////////////////////////

!sycl_group_1_ = type !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>
!sycl_group_2_ = type !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>

// CHECK: func @_Z7group_1N2cl4sycl5groupILi1EEE(%arg0: !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>)
func @_Z7group_1N2cl4sycl5groupILi1EEE(%arg0: !sycl_group_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z7group_2N2cl4sycl5groupILi2EEE(%arg0: !sycl.group<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>)
func @_Z7group_2N2cl4sycl5groupILi2EEE(%arg0: !sycl_group_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

