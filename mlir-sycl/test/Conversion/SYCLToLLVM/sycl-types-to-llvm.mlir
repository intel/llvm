// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
// CHECK: llvm.func @test_array.1(%arg0: !llvm.[[ARRAY_1:struct<"class.sycl::_V1::detail::array.*", \(array<1 x i64>\)>]])
func.func @test_array.1(%arg0: !sycl_array_1_) {
  return
}
// CHECK: llvm.func @test_array.2(%arg0: !llvm.[[ARRAY_2:struct<"class.sycl::_V1::detail::array.*", \(array<2 x i64>\)>]])
func.func @test_array.2(%arg0: !sycl_array_2_) {
  return
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @test_id(%arg0: !llvm.[[ID_1:struct<"class.sycl::_V1::id.*", \(]][[ARRAY_1]][[SUFFIX:\)>]], %arg1: !llvm.[[ID_1]][[ARRAY_1]][[SUFFIX]])
func.func @test_id(%arg0: !sycl_id_1_, %arg1: !sycl_id_1_) {
  return
}

// -----

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @test_range.1(%arg0: !llvm.[[RANGE_1:struct<"class.sycl::_V1::range.*", \(]][[ARRAY_1]][[SUFFIX]])
func.func @test_range.1(%arg0: !sycl_range_1_) {
  return
}
// CHECK: llvm.func @test_range.2(%arg0: !llvm.[[RANGE_2:struct<"class.sycl::_V1::range.*", \(]][[ARRAY_2]][[SUFFIX]])
func.func @test_range.2(%arg0: !sycl_range_2_) {
  return
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_1_f32_rw_gb = !sycl.accessor<[1, f32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<f32, 1>)>)>
!sycl_accessor_2_f32_rw_gb = !sycl.accessor<[2, f32, read_write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(ptr<f32, 1>)>)>
!sycl_LocalAccessorBaseDevice_1_ = !sycl.LocalAccessorBaseDevice<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_local_accessor_base_1_i32_rw = !sycl.local_accessor_base<[1, i32, read_write], (!sycl_LocalAccessorBaseDevice_1_, memref<?xi32, 3>)>
!sycl_accessor_1_i32_rw_1 = !sycl.accessor<[1, i32, read_write, local], (!sycl_local_accessor_base_1_i32_rw)>
!sycl_accessor_subscript_1_ = !sycl.accessor_subscript<[1], (!sycl_id_2_, !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(ptr<i32, 1>)>)>)>
// CHECK: llvm.func @test_accessor_common(%arg0: !llvm.struct<"class.sycl::_V1::detail::accessor_common", (i8)>)
func.func @test_accessor_common(%arg0: !sycl.accessor_common) {
  return
}
// CHECK: llvm.func @test_accessorImplDevice(%arg0: !llvm.[[ACCESSORIMPLDEVICE_1:struct<"class.sycl::_V1::detail::AccessorImplDevice.*", \(]][[ID_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
func.func @test_accessorImplDevice(%arg0: !sycl_accessor_impl_device_1_) {
  return
}
// CHECK: llvm.func @test_accessor.1(%arg0: !llvm.struct<"class.sycl::_V1::accessor{{.*}}", ([[ACCESSORIMPLDEVICE_1]][[ID_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]], struct<(ptr<i32, 1>)>[[SUFFIX]])
func.func @test_accessor.1(%arg0: !sycl_accessor_1_i32_rw_gb) {
  return
}
// CHECK: llvm.func @test_accessor.2(%arg0: !llvm.[[ACCESSOR_2:struct<"class.sycl::_V1::accessor.*", \(]][[ACCESSORIMPLDEVICE_2:struct<"class.sycl::_V1::detail::AccessorImplDevice.*", \(]][[ID_2:struct<"class.sycl::_V1::id.*", \(]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]], struct<(ptr<i32, 1>)>[[SUFFIX]])
func.func @test_accessor.2(%arg0: !sycl_accessor_2_i32_rw_gb) {
  return
}
// CHECK: llvm.func @test_accessor.3(%arg0: !llvm.struct<"class.sycl::_V1::accessor{{.*}}", ([[ACCESSORIMPLDEVICE_1]][[ID_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]], struct<(ptr<f32, 1>)>[[SUFFIX]])
func.func @test_accessor.3(%arg0: !sycl_accessor_1_f32_rw_gb) {
  return
}
// CHECK: llvm.func @test_accessor.4(%arg0: !llvm.struct<"class.sycl::_V1::accessor{{.*}}", ([[ACCESSORIMPLDEVICE_2]][[ID_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]], struct<(ptr<f32, 1>)>[[SUFFIX]])
func.func @test_accessor.4(%arg0: !sycl_accessor_2_f32_rw_gb) {
  return
}
// CHECK: llvm.func @test_accessor.5(%arg0: !llvm.struct<"class.sycl::_V1::accessor{{.*}}", (struct<"class.sycl::_V1::local_accessor_base{{.*}}", ([[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]], ptr<i32, 3>
func.func @test_accessor.5(%arg0: !sycl_accessor_1_i32_rw_1) {
  return
}
// CHECK: llvm.func @test_accessorSubscript(%arg0: !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript{{.*}}", ([[ID_2]][[ARRAY_2]][[SUFFIX]], [[ACCESSOR_2]][[ACCESSORIMPLDEVICE_2]][[ID_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]], struct<(ptr<i32, 1>)>[[SUFFIX]]) 
func.func @test_accessorSubscript(%arg0: !sycl_accessor_subscript_1_) {
  return
}
// CHECK: llvm.func @test_OwnerLessBase(%arg0: !llvm.struct<"class.sycl::_V1::detail::OwnerLessBase", (i8)>)
func.func @test_OwnerLessBase(%arg0: !sycl.owner_less_base) {
  return
}

// -----

!sycl_atomic_i32_glo = !sycl.atomic<[i32, global], (memref<?xi32, 1>)>
!sycl_atomic_f32_loc = !sycl.atomic<[f32, local], (memref<?xf32, 3>)>
// CHECK: llvm.func @test_atomic(%arg0: !llvm.[[ATOMIC1:struct<"class.sycl::_V1::atomic", \(ptr<f32, 3>\)>]], %arg1: !llvm.[[ATOMIC1:struct<"class.sycl::_V1::atomic.1", \(ptr<i32, 1>\)>]]) {
func.func @test_atomic(%arg0: !sycl_atomic_f32_loc, %arg1: !sycl_atomic_i32_glo) {
  return
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_h_item_1_ = !sycl.h_item<[1], (!sycl_item_1_1_, !sycl_item_1_1_, !sycl_item_1_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>
// CHECK: llvm.func @test_itemBase.true(%arg0: !llvm.[[ITEM_BASE_1_TRUE:struct<"struct.sycl::_V1::detail::ItemBase.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
func.func @test_itemBase.true(%arg0: !sycl_item_base_1_) {
  return
}
// CHECK: llvm.func @test_itemBase.false(%arg0: !llvm.[[ITEM_BASE_1_FALSE:struct<"struct.sycl::_V1::detail::ItemBase.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
func.func @test_itemBase.false(%arg0: !sycl_item_base_1_1) {
  return
}
// CHECK: llvm.func @test_item.true(%arg0: !llvm.[[ITEM_1_TRUE:struct<"class.sycl::_V1::item.*", \(]][[ITEM_BASE_1_TRUE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]])
func.func @test_item.true(%arg0: !sycl_item_1_) {
  return
}
// CHECK: llvm.func @test_item.false(%arg0: !llvm.[[ITEM_1_FALSE:struct<"class.sycl::_V1::item.*", \(]][[ITEM_BASE_1_FALSE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]])
func.func @test_item.false(%arg0: !sycl_item_1_1_) {
  return
}
// CHECK: llvm.func @test_group(%arg0: !llvm.[[GROUP:struct<"class.sycl::_V1::group.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
func.func @test_group(%arg0: !sycl_group_1_) {
  return
}
// CHECK: llvm.func @test_h_item(%arg0: !llvm.[[H_ITEM:struct<"class.sycl::_V1::h_item", \(]][[ITEM_1_FALSE]][[ITEM_BASE_1_FALSE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]], [[ITEM_1_FALSE]][[ITEM_BASE_1_FALSE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]], [[ITEM_1_FALSE]][[ITEM_BASE_1_FALSE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]][[SUFFIX]]) {
func.func @test_h_item(%arg0: !sycl_h_item_1_) {
  return
}
// CHECK: llvm.func @test_nd_item(%arg0: !llvm.[[ND_ITEM_1:struct<"class.sycl::_V1::nd_item.*", \(]][[ITEM_1_TRUE]][[ITEM_BASE_1_TRUE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]], [[ITEM_1_FALSE]][[ITEM_BASE_1_FALSE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]], [[GROUP]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]])
func.func @test_nd_item(%arg0: !sycl_nd_item_1_) {
  return
}

// -----

!sycl_kernel_handler_ = !sycl.kernel_handler<(!llvm.ptr<i8, 4>)>
// CHECK: llvm.func @test_kernel_handler(%arg0: !llvm.[[KERNEL_HANDLER:struct<"class.sycl::_V1::kernel_handler", \(ptr<i8, 4>\)>]]) {
func.func @test_kernel_handler(%arg0: !sycl_kernel_handler_) {
  return
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_LocalAccessorBaseDevice_1_ = !sycl.LocalAccessorBaseDevice<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_local_accessor_base_1_i32_rw = !sycl.local_accessor_base<[1, i32, read_write], (!sycl_LocalAccessorBaseDevice_1_, memref<?xi32, 3>)>
!sycl_local_accessor_1_i32_ = !sycl.local_accessor<[1, i32], (!sycl_local_accessor_base_1_i32_rw)>

// CHECK: llvm.func @test_local_accessor_base_device(%arg0: !llvm.[[LOCAL_ACCESSOR_BASE_DEVICE:struct<"class.sycl::_V1::detail::LocalAccessorBaseDevice.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]])
func.func @test_local_accessor_base_device(%arg0: !sycl_LocalAccessorBaseDevice_1_) {
  return
}
// CHECK: llvm.func @test_local_accessor_base(%arg0: !llvm.[[LOCAL_ACCESSOR_BASE:struct<"class.sycl::_V1::local_accessor_base.*", \(]][[LOCAL_ACCESSOR_BASE_DEVICE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]])
func.func @test_local_accessor_base(%arg0: !sycl_local_accessor_base_1_i32_rw) {
  return
}
// CHECK: llvm.func @test_local_accessor(%arg0: !llvm.[[LOCAL_ACCESSOR:struct<"class.sycl::_V1::local_accessor.*", \(]][[LOCAL_ACCESSOR_BASE]][[LOCAL_ACCESSOR_BASE_DEVICE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]], ptr<i32, 3>
func.func @test_local_accessor(%arg0: !sycl_local_accessor_1_i32_) {
  return
}

// -----

!sycl_maximum_i32_ = !sycl.maximum<i32>
!sycl_minimum_i32_ = !sycl.minimum<i32>
// CHECK: llvm.func @test_maximum(%arg0: !llvm.[[MAXIMUM:struct<"struct.sycl::_V1::maximum", \(i8\)>]]) {
func.func @test_maximum(%arg0: !sycl_maximum_i32_) {
  return
}
// CHECK: llvm.func @test_minimum(%arg0: !llvm.[[MINIMUM:struct<"struct.sycl::_V1::minimum", \(i8\)>]]) {
func.func @test_minimum(%arg0: !sycl_minimum_i32_) {
  return
}

// -----

!sycl_multi_ptr_i32_glo = !sycl.multi_ptr<[i32, global, yes], (memref<?xi32, 1>)>
// CHECK: llvm.func @test_multi_ptr(%arg0: !llvm.[[ATOMIC1:struct<"class.sycl::_V1::multi_ptr", \(ptr<i32, 1>\)>]]) {
func.func @test_multi_ptr(%arg0: !sycl_multi_ptr_i32_glo) {
    return
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_range_2_ = !sycl.nd_range<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
// CHECK: llvm.func @test_nd_range.1(%arg0: !llvm.[[ND_RANGE_1:struct<"class.sycl::_V1::nd_range.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]]) {
func.func @test_nd_range.1(%arg0: !sycl_nd_range_1_) {
  return
}
// CHECK: llvm.func @test_nd_range.2(%arg0: !llvm.[[ND_RANGE_2:struct<"class.sycl::_V1::nd_range.*", \(]][[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[ID_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]]) {
func.func @test_nd_range.2(%arg0: !sycl_nd_range_2_) {
  return
}

// -----

// CHECK: llvm.func @test_sub_group(%arg0: !llvm.[[SUB_GROUP:struct<"struct.sycl::_V1::ext::oneapi::sub_group", \(i8\)>]]) {
func.func @test_sub_group(%arg0: !sycl.sub_group) {
  return
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_1_i8_rw_gb = !sycl.accessor<[1, i8, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i8, 1>)>)>
!sycl_stream_ = !sycl.stream<(!llvm.array<16 x i8>, !sycl_accessor_1_i8_rw_gb, !sycl_accessor_1_i32_ato_gb, !sycl_accessor_1_i8_rw_gb, i32, i64, i32, i32, i32, i32)>
// CHECK: llvm.func @test_stream(%arg0: !llvm.struct<"class.sycl::_V1::stream", (array<16 x i8>, struct<"class.sycl::_V1::accessor.1", (struct<"class.sycl::_V1::detail::AccessorImplDevice.1", (struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>)>, struct<(ptr<i8, 1>)>)>, struct<"class.sycl::_V1::accessor.1.1", (struct<"class.sycl::_V1::detail::AccessorImplDevice.1", (struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>)>, struct<(ptr<i32, 1>)>)>, struct<"class.sycl::_V1::accessor.1", (struct<"class.sycl::_V1::detail::AccessorImplDevice.1", (struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>)>, struct<(ptr<i8, 1>)>)>, i32, i64, i32, i32, i32, i32)>) {
func.func @test_stream(%arg0: !sycl_stream_) {
    return
}
// -----

!sycl_vec_f32_4_ = !sycl.vec<[f32, 4], (vector<4xf32>)>
// CHECK: llvm.func @test_vec(%arg0: !llvm.[[VEC:struct<"class.sycl::_V1::vec", \(vector<4xf32>\)>]])
func.func @test_vec(%arg0: !sycl_vec_f32_4_) {
  return
}
!sycl_swizzled_vec_f32_4_ = !sycl.swizzled_vec<[!sycl_vec_f32_4_, 0, 2], (memref<?x!sycl_vec_f32_4_, 4>, !llvm.struct<(i8)>, !llvm.struct<(i8)>)>
// CHECK: llvm.func @test_swizzled_vec(%arg0: !llvm.[[SWIZZLED_VEC:struct<"class.sycl::_V1::detail::SwizzleOp"]], (ptr<[[VEC]], 4>, [[GET_OP:struct<\(i8\)>]], [[GET_OP]][[SUFFIX]]) {
func.func @test_swizzled_vec(%arg0: !sycl_swizzled_vec_f32_4_) {
  return
}

// -----
