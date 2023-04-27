// Verify the type aliases printed output can be parsed.
// RUN: sycl-mlir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// Verify the generic printed output can be parsed.
// RUN: sycl-mlir-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-local-scope | sycl-mlir-opt -allow-unregistered-dialect | FileCheck %s

////////////////////////////////////////////////////////////////////////////////
// ID
////////////////////////////////////////////////////////////////////////////////

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: func @_Z4id_1N2cl4sycl2idILi1EEE(%arg0: !sycl_id_1_)
func.func @_Z4id_1N2cl4sycl2idILi1EEE(%arg0: !sycl_id_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z4id_2N2cl4sycl2idILi2EEE(%arg0: !sycl_id_2_)
func.func @_Z4id_2N2cl4sycl2idILi2EEE(%arg0: !sycl_id_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// ACCESSOR
////////////////////////////////////////////////////////////////////////////////

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_accessor_1_i32_w_gb = !sycl.accessor<[1, i32, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_2_i32_r_gb = !sycl.accessor<[2, i32, read, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK: func @_Z5acc_1N2cl4sycl8accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl_accessor_1_i32_w_gb)
func.func @_Z5acc_1N2cl4sycl8accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl_accessor_1_i32_w_gb) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z5acc_2N2cl4sycl8accessorIiLi2ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl_accessor_2_i32_r_gb)
func.func @_Z5acc_2N2cl4sycl8accessorIiLi2ELNS0_6access4modeE1024ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE(%arg0: !sycl_accessor_2_i32_r_gb) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// RANGE
////////////////////////////////////////////////////////////////////////////////

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: func @_Z7range_1N2cl4sycl5rangeILi1EEE(%arg0: !sycl_range_1_)
func.func @_Z7range_1N2cl4sycl5rangeILi1EEE(%arg0: !sycl_range_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z7range_2N2cl4sycl5rangeILi2EEE(%arg0: !sycl_range_2_)
func.func @_Z7range_2N2cl4sycl5rangeILi2EEE(%arg0: !sycl_range_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// ND_RANGE
////////////////////////////////////////////////////////////////////////////////

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_range_2_ = !sycl.nd_range<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>

// CHECK: func @_Z10nd_range_1N4sycl3_V18nd_rangeILi1EEE(%arg0: !sycl_nd_range_1_)
func.func @_Z10nd_range_1N4sycl3_V18nd_rangeILi1EEE(%arg0: !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z10nd_range_2N4sycl3_V18nd_rangeILi2EEE(%arg0: !sycl_nd_range_2_)
func.func @_Z10nd_range_2N4sycl3_V18nd_rangeILi2EEE(%arg0: !sycl.nd_range<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// ARRAY
////////////////////////////////////////////////////////////////////////////////

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>

// CHECK: func @_Z5arr_1N2cl4sycl6detail5arrayILi1EEE(%arg0: !sycl_array_1_)
func.func @_Z5arr_1N2cl4sycl6detail5arrayILi1EEE(%arg0: !sycl_array_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z5arr_2N2cl4sycl6detail5arrayILi2EEE(%arg0: !sycl_array_2_)
func.func @_Z5arr_2N2cl4sycl6detail5arrayILi2EEE(%arg0: !sycl_array_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// ITEM
////////////////////////////////////////////////////////////////////////////////

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>

// CHECK: func @_Z11item_1_trueN2cl4sycl4itemILi1ELb1EEE(%arg0: !sycl_item_1_)
func.func @_Z11item_1_trueN2cl4sycl4itemILi1ELb1EEE(%arg0: !sycl_item_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z12item_2_falseN2cl4sycl4itemILi2ELb0EEE(%arg0: !sycl_item_2_)
func.func @_Z12item_2_falseN2cl4sycl4itemILi2ELb0EEE(%arg0: !sycl_item_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// ND_ITEM 
////////////////////////////////////////////////////////////////////////////////

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>)>, !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>)>, !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>)>
!sycl_nd_item_2_ = !sycl.nd_item<[2], (!sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>)>, !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>, !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>)>

// CHECK: func @_Z9nd_item_1N2cl4sycl7nd_itemILi1EEE(%arg0: !sycl_nd_item_1_)
func.func @_Z9nd_item_1N2cl4sycl7nd_itemILi1EEE(%arg0: !sycl_nd_item_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z9nd_item_2N2cl4sycl7nd_itemILi2EEE(%arg0: !sycl_nd_item_2_)
func.func @_Z9nd_item_2N2cl4sycl7nd_itemILi2EEE(%arg0: !sycl_nd_item_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// GROUP
////////////////////////////////////////////////////////////////////////////////

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>

// CHECK: func @_Z7group_1N2cl4sycl5groupILi1EEE(%arg0: !sycl_group_1_)
func.func @_Z7group_1N2cl4sycl5groupILi1EEE(%arg0: !sycl_group_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @_Z7group_2N2cl4sycl5groupILi2EEE(%arg0: !sycl_group_2_)
func.func @_Z7group_2N2cl4sycl5groupILi2EEE(%arg0: !sycl_group_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// VEC
////////////////////////////////////////////////////////////////////////////////

!sycl_vec_i32_2_ = !sycl.vec<[i32, 2], (vector<2xi32>)>
!sycl_vec_f32_4_ = !sycl.vec<[f32, 4], (vector<4xf32>)>

// CHECK: func @vec_0(%arg0: !sycl_vec_i32_2_)
func.func @vec_0(%arg0: !sycl_vec_i32_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @vec_1(%arg0: !sycl_vec_f32_4_)
func.func @vec_1(%arg0: !sycl_vec_f32_4_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

////////////////////////////////////////////////////////////////////////////////
// SWIZZLED_VEC
////////////////////////////////////////////////////////////////////////////////

!sycl_swizzled_vec_i32_2_ = !sycl.swizzled_vec<[!sycl_vec_i32_2_, 0, 1], (memref<?x!sycl_vec_i32_2_, 4>, !llvm.struct<(i8)>, !llvm.struct<(i8)>)>
!sycl_swizzled_vec_i32_2_1 = !sycl.swizzled_vec<[!sycl_vec_i32_2_, 0, 1], (memref<?x!sycl_vec_i32_2_, 4>, !sycl_swizzled_vec_i32_2_, !llvm.struct<(i32)>)>

// CHECK: func @swizzled_vec_0(%arg0: !sycl_swizzled_vec_i32_2_)
func.func @swizzled_vec_0(%arg0: !sycl_swizzled_vec_i32_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
// CHECK: func @swizzled_vec_1(%arg0: !sycl_swizzled_vec_i32_2_1)
func.func @swizzled_vec_1(%arg0: !sycl_swizzled_vec_i32_2_1) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

// -----

////////////////////////////////////////////////////////////////////////////////
// ATOMIC
////////////////////////////////////////////////////////////////////////////////

!sycl_atomic_f32_loc = !sycl.atomic<[f32, local], (memref<?xf32, 3>)>
!sycl_atomic_i32_glo = !sycl.atomic<[i32, global], (memref<?xi32, 1>)>

// CHECK: func @_Z6atomicN4sycl3_V16atomicIiLNS0_6access13address_spaceE1EEE(%arg0: !sycl_atomic_f32_loc, %arg1: !sycl_atomic_i32_glo)
func.func @_Z6atomicN4sycl3_V16atomicIiLNS0_6access13address_spaceE1EEE(%arg0: !sycl_atomic_f32_loc, %arg1: !sycl_atomic_i32_glo) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}


////////////////////////////////////////////////////////////////////////////////
// multi_ptr
////////////////////////////////////////////////////////////////////////////////

!sycl_multi_ptr_i32_glo = !sycl.multi_ptr<[i32, global, yes], (memref<?xi32, 1>)>

// CHECK: func @_Z9multi_ptrN4sycl3_V19multi_ptrIiLNS0_6access13address_spaceE1ELNS2_9decoratedE1EEE(%arg0: !sycl_multi_ptr_i32_glo)
func.func @_Z9multi_ptrN4sycl3_V19multi_ptrIiLNS0_6access13address_spaceE1ELNS2_9decoratedE1EEE(%arg0: !sycl_multi_ptr_i32_glo) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}

////////////////////////////////////////////////////////////////////////////////
// stream
////////////////////////////////////////////////////////////////////////////////

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_1_i8_rw_gb = !sycl.accessor<[1, i8, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i8, 1>)>)>
!sycl_stream_ = !sycl.stream<(!llvm.array<16 x i8>, !sycl_accessor_1_i8_rw_gb, !sycl_accessor_1_i32_ato_gb, !sycl_accessor_1_i8_rw_gb, i32, i64, i32, i32, i32, i32)>

// CHECK: func @_Z6streamN4sycl3_V16streamE(%arg0: !sycl_stream_)
func.func @_Z6streamN4sycl3_V16streamE(%arg0: !sycl_stream_) attributes {llvm.linkage = #llvm.linkage<external>} {
  return
}
