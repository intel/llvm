// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.func @test_array.1(%arg0: !llvm.[[ARRAY_1:struct<"class.sycl::_V1::detail::array.*", \(array<1 x i64>\)>]])
// CHECK: llvm.func @test_array.2(%arg0: !llvm.[[ARRAY_2:struct<"class.sycl::_V1::detail::array.*", \(array<2 x i64>\)>]])
// CHECK: llvm.func @test_id(%arg0: !llvm.[[ID_1:struct<"class.sycl::_V1::id.*", \(]][[ARRAY_1]][[SUFFIX:\)>]], %arg1: !llvm.[[ID_1]][[ARRAY_1]][[SUFFIX]])
// CHECK: llvm.func @test_range.1(%arg0: !llvm.[[RANGE_1:struct<"class.sycl::_V1::range.*", \(]][[ARRAY_1]][[SUFFIX]])
// CHECK: llvm.func @test_range.2(%arg0: !llvm.[[RANGE_2:struct<"class.sycl::_V1::range.*", \(]][[ARRAY_2]][[SUFFIX]])
// CHECK: llvm.func @test_nd_range.1(%arg0: !llvm.[[ND_RANGE_1:struct<"class.sycl::_V1::nd_range.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]]) {
// CHECK: llvm.func @test_nd_range.2(%arg0: !llvm.[[ND_RANGE_2:struct<"class.sycl::_V1::nd_range.*", \(]][[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[ID_2:struct<"class.sycl::_V1::id.*", \(]][[ARRAY_2]][[SUFFIX]][[SUFFIX]]) {
// CHECK: llvm.func @test_accessorImplDevice(%arg0: !llvm.[[ACCESSORIMPLDEVICE_1:struct<"class.sycl::_V1::detail::AccessorImplDevice.*", \(]][[ID_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_accessor_common(%arg0: !llvm.[[ACCESSOR_COMMON:struct<"class.sycl::_V1::detail::accessor_common", \(i8\)>]])
// CHECK: llvm.func @test_accessor.1(%arg0: !llvm.[[ACCESSOR_1:struct<"class.sycl::_V1::accessor.*", \(]][[ACCESSORIMPLDEVICE_1]][[ID_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]], struct<(ptr<i32, 1>)>[[SUFFIX]])
// CHECK: llvm.func @test_accessor.2(%arg0: !llvm.[[ACCESSOR_2:struct<"class.sycl::_V1::accessor.*", \(]][[ACCESSORIMPLDEVICE_2:struct<"class.sycl::_V1::detail::AccessorImplDevice.*", \(]][[ID_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]], struct<(ptr<i32, 1>)>[[SUFFIX]])
// CHECK: llvm.func @test_accessor.3(%arg0: !llvm.[[ACCESSOR_3:struct<"class.sycl::_V1::accessor.*", \(]][[ACCESSORIMPLDEVICE_1]][[ID_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]], struct<(ptr<f32, 1>)>[[SUFFIX]])
// CHECK: llvm.func @test_accessor.4(%arg0: !llvm.[[ACCESSOR_4:struct<"class.sycl::_V1::accessor.*", \(]][[ACCESSORIMPLDEVICE_2]][[ID_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]], struct<(ptr<f32, 1>)>[[SUFFIX]])
// CHECK: llvm.func @test_accessorSubscript(%arg0: !llvm.[[ACCESSORSUBSCRIPT_1:struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.*", \(]][[ID_2]][[ARRAY_2]][[SUFFIX]], [[ACCESSOR_2]][[ACCESSORIMPLDEVICE_2]][[ID_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]], [[RANGE_2]][[ARRAY_2]][[SUFFIX]][[SUFFIX]], struct<(ptr<i32, 1>)>[[SUFFIX]]) 
// CHECK: llvm.func @test_itemBase.true(%arg0: !llvm.[[ITEM_BASE_1_TRUE:struct<"struct.sycl::_V1::detail::ItemBase.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_itemBase.false(%arg0: !llvm.[[ITEM_BASE_1_FALSE:struct<"struct.sycl::_V1::detail::ItemBase.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_item.true(%arg0: !llvm.[[ITEM_1_TRUE:struct<"class.sycl::_V1::item.*", \(]][[ITEM_BASE_1_TRUE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_item.false(%arg0: !llvm.[[ITEM_1_FALSE:struct<"class.sycl::_V1::item.*", \(]][[ITEM_BASE_1_FALSE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_group(%arg0: !llvm.[[GROUP_1:struct<"class.sycl::_V1::group.*", \(]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_get_op(%arg0: !llvm.[[GETOP:struct<"class.sycl::_V1::detail::GetOp", \(i8\)>]])
// CHECK: llvm.func @test_get_scalar_op(%arg0: !llvm.[[GETSCALAROP:struct<"class.sycl::_V1::detail::GetScalarOp.*", \(i32\)>]])
// CHECK: llvm.func @test_nd_item(%arg0: !llvm.[[ND_ITEM_1:struct<"class.sycl::_V1::nd_item.*", \(]][[ITEM_1_TRUE]][[ITEM_BASE_1_TRUE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]], [[ITEM_1_FALSE]][[ITEM_BASE_1_FALSE]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]], [[GROUP_1]][[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[RANGE_1]][[ARRAY_1]][[SUFFIX]], [[ID_1]][[ARRAY_1]][[SUFFIX]][[SUFFIX]][[SUFFIX]])
// CHECK: llvm.func @test_vec(%arg0: !llvm.[[VEC:struct<"class.sycl::_V1::vec", \(vector<4xf32>\)>]])
// CHECK: llvm.func @test_atomic(%arg0: !llvm.[[ATOMIC1:struct<"class.sycl::_V1::atomic", \(struct<\(ptr<f32, 3>, ptr<f32, 3>, i64, array<1 x i64>, array<1 x i64>\)>\)>]], %arg1: !llvm.[[ATOMIC1:struct<"class.sycl::_V1::atomic.1", \(struct<\(ptr<i32, 1>, ptr<i32, 1>, i64, array<1 x i64>, array<1 x i64>\)>\)>]]) {


!sycl_array_1 = !sycl.array<[1], (memref<1xi64>)>
!sycl_array_2 = !sycl.array<[2], (memref<2xi64>)>
!sycl_nd_range_1 = !sycl.nd_range<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>
!sycl_nd_range_2 = !sycl.nd_range<[2], (!sycl.range<2>, !sycl.range<2>, !sycl.id<2>)>
!sycl_accessor_impl_device_1 = !sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>
!sycl_accessor_1_i32 = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_2_i32 = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl_accessor_impl_device_2, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_1_f32 = !sycl.accessor<[1, f32, read_write, global_buffer], (!sycl_accessor_impl_device_1, !llvm.struct<(ptr<f32, 1>)>)>
!sycl_accessor_2_f32 = !sycl.accessor<[2, f32, read_write, global_buffer], (!sycl_accessor_impl_device_2, !llvm.struct<(ptr<f32, 1>)>)>
!sycl_accessor_subscript_1 = !sycl.accessor_subscript<[1], (!sycl.id<2>, !sycl_accessor_2_i32)>
!sycl_item_base_1_true = !sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>
!sycl_item_base_1_false = !sycl.item_base<[1, false], (!sycl.range<1>, !sycl.id<1>)>
!sycl_item_1_true = !sycl.item<[1, true], (!sycl_item_base_1_true)>
!sycl_item_1_false = !sycl.item<[1, false], (!sycl_item_base_1_false)>
!sycl_group_1 = !sycl.group<[1], (!sycl.range<1>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)>
!sycl_nd_item_1 = !sycl.nd_item<[1], (!sycl_item_1_true, !sycl_item_1_false, !sycl_group_1)>
!sycl_atomic_f32_3_ = !sycl.atomic<[f32,3], (memref<?xf32, 3>)>
!sycl_atomic_i32_1_ = !sycl.atomic<[i32,1], (memref<?xi32, 1>)>

module {
  func.func @test_array.1(%arg0: !sycl_array_1) {
    return
  }
  func.func @test_array.2(%arg0: !sycl_array_2) {
    return
  }
  func.func @test_id(%arg0: !sycl.id<1>, %arg1: !sycl.id<1>) {
    return
  }
  func.func @test_range.1(%arg0: !sycl.range<1>) {
    return
  }
  func.func @test_range.2(%arg0: !sycl.range<2>) {
    return
  }
  func.func @test_nd_range.1(%arg0: !sycl_nd_range_1) {
    return
  }
  func.func @test_nd_range.2(%arg0: !sycl_nd_range_2) {
    return
  }
  func.func @test_accessorImplDevice(%arg0: !sycl_accessor_impl_device_1) {
    return
  }
  func.func @test_accessor_common(%arg0: !sycl.accessor_common) {
    return
  }
  func.func @test_accessor.1(%arg0: !sycl_accessor_1_i32) {
    return
  }
  func.func @test_accessor.2(%arg0: !sycl_accessor_2_i32) {
    return
  }
  func.func @test_accessor.3(%arg0: !sycl_accessor_1_f32) {
    return
  }
  func.func @test_accessor.4(%arg0: !sycl_accessor_2_f32) {
    return
  }
  func.func @test_accessorSubscript(%arg0: !sycl_accessor_subscript_1) {
    return
  }
  func.func @test_itemBase.true(%arg0: !sycl_item_base_1_true) {
    return
  }
  func.func @test_itemBase.false(%arg0: !sycl_item_base_1_false) {
    return
  }
  func.func @test_item.true(%arg0: !sycl_item_1_true) {
    return
  }
  func.func @test_item.false(%arg0: !sycl_item_1_false) {
    return
  }
  func.func @test_group(%arg0: !sycl_group_1) {
    return
  }
  func.func @test_get_op(%arg0: !sycl.get_op) {
    return
  }
  func.func @test_get_scalar_op(%arg0: !sycl.get_scalar_op<[i32], (i32)>) {
    return
  }
  func.func @test_nd_item(%arg0: !sycl_nd_item_1) {
    return
  }
  func.func @test_vec(%arg0: !sycl.vec<[f32, 4], (vector<4xf32>)>) {
    return
  }
  func.func @test_atomic(%arg0: !sycl_atomic_f32_3_, %arg1: !sycl_atomic_i32_1_) {
    return
  }
}
