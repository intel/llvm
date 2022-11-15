// RUN: sycl-mlir-opt -sycl-method-to-sycl-call -split-input-file -verify-diagnostics %s | FileCheck %s

!sycl_accessor_2_i32_read_write_global_buffer = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_1_ = !sycl.id<1>
!sycl_id_2_ = !sycl.id<2>
!sycl_item_1_1_ = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>
!sycl_range_2_ = !sycl.range<2>

// CHECK-LABEL: func.func @accessor_subscript_operator(%arg0: memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, %arg1: memref<?x!sycl_id_2_>) -> memref<?xi32, 4> {
// CHECK-NEXT: %0 = sycl.call(%arg0, %arg1) {FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEERiNS0_2idILi2EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, memref<?x!sycl_id_2_>) -> memref<?xi32, 4>

func.func @accessor_subscript_operator(%arg0: memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, %arg1: memref<?x!sycl_id_2_>) -> memref<?xi32, 4> {
  %0 = sycl.accessor.subscript %arg0[%arg1] {BaseType = memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEERiNS0_2idILi2EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, memref<?x!sycl_id_2_>) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// CHECK-LABEL: func.func @range_get(%arg0: memref<?x!sycl_range_2_, 4>, %arg1: i32) -> i64 {
// CHECK-NEXT: %0 = sycl.cast(%arg0) : (memref<?x!sycl_range_2_, 4>) -> memref<?x!sycl_array_2_, 4>
// CHECK-NEXT: %1 = sycl.call(%0, %arg1) {FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, TypeName = @array} : (memref<?x!sycl_array_2_, 4>, i32) -> i64

func.func @range_get(%arg0: memref<?x!sycl_range_2_, 4>, %arg1: i32) -> i64 {
  %0 = "sycl.range.get"(%arg0, %arg1) {BaseType = memref<?x!sycl_array_2_, 4>, FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, TypeName = @array} : (memref<?x!sycl_range_2_, 4>, i32) -> i64
  return %0 : i64
}

// CHECK-LABEL: func.func @range_size(%arg0: memref<?x!sycl_range_2_, 4>) -> i64 {
// CHECK-NEXT: %0 = sycl.call(%arg0) {FunctionName = @size, MangledFunctionName = @_ZNK4sycl3_V15rangeILi2EE4sizeEv, TypeName = @range} : (memref<?x!sycl_range_2_, 4>) -> i64

func.func @range_size(%arg0: memref<?x!sycl_range_2_, 4>) -> i64 {
  %0 = "sycl.range.size"(%arg0) {BaseType = memref<?x!sycl_range_2_, 4>, FunctionName = @size, MangledFunctionName = @_ZNK4sycl3_V15rangeILi2EE4sizeEv, TypeName = @range} : (memref<?x!sycl_range_2_, 4>) -> i64
  return %0 : i64
}

// CHECK-LABEL: func.func @sycl_item_get_id(%arg0: memref<?x!sycl_item_1_1_, 4>) -> !sycl_id_1_ {
// CHECK-NEXT: %0 = sycl.call(%arg0) {FunctionName = @get_id, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE6get_idEv, TypeName = @item} : (memref<?x!sycl_item_1_1_, 4>) -> !sycl_id_1_

func.func @sycl_item_get_id(%arg0: memref<?x!sycl_item_1_1_, 4>) -> !sycl_id_1_ {
  %0 = "sycl.item.get_id"(%arg0) {BaseType = memref<?x!sycl_item_1_1_, 4>, FunctionName = @get_id, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE6get_idEv, TypeName = @item} : (memref<?x!sycl_item_1_1_, 4>) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}
