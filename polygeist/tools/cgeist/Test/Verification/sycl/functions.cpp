// Copyright (C) Codeplay Software Limited

//===--- functions.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: clang++ -fsycl -fsycl-device-only -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: !sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK-MLIR: !sycl_accessor_2_i32_read_write_global_buffer = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK-MLIR: !sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
// CHECK-MLIR: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>

// CHECK-LLVM: %"class.sycl::_V1::accessor.2" = type { %"class.sycl::_V1::detail::AccessorImplDevice.2", { i32 addrspace(1)* } }
// CHECK-LLVM: %"class.sycl::_V1::detail::AccessorImplDevice.2" = type { %"class.sycl::_V1::id.2", %"class.sycl::_V1::range.2", %"class.sycl::_V1::range.2" }
// CHECK-LLVM: %"class.sycl::_V1::range.2" = type { %"class.sycl::_V1::detail::array.2" }
// CHECK-LLVM: %"class.sycl::_V1::detail::array.2" = type { [2 x i64] }
// CHECK-LLVM: %"class.sycl::_V1::id.2" = type { %"class.sycl::_V1::detail::array.2" }
// CHECK-LLVM: %"class.sycl::_V1::accessor.1" = type { %"class.sycl::_V1::detail::AccessorImplDevice.1", { i32 addrspace(1)* } }
// CHECK-LLVM: %"class.sycl::_V1::detail::AccessorImplDevice.1" = type { %"class.sycl::_V1::id.1", %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1" }
// CHECK-LLVM: %"class.sycl::_V1::id.1" = type { %"class.sycl::_V1::detail::array.1" }
// CHECK-LLVM: %"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
// CHECK-LLVM: %"class.sycl::_V1::range.1" = type { %"class.sycl::_V1::detail::array.1" }

template <typename T> SYCL_EXTERNAL void keep(T);

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_0N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEENS0_2idILi2EEE(%{{.*}}: !sycl_accessor_2_i32_read_write_global_buffer, %{{.*}}: !sycl_id_2_) attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] {BaseType = memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEERiNS0_2idILi2EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, !sycl_id_2_) -> memref<?xi32, 4>

// CHECK-LLVM: define spir_func void @_Z29accessor_subscript_operator_0N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEENS0_2idILi2EEE(%"class.sycl::_V1::accessor.2" %{{.*}}, %"class.sycl::_V1::id.2" %{{.*}}) #0 {
// CHECK-LLVM:  %{{.*}} = call i32 addrspace(4)* @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEERiNS0_2idILi2EEE(%"class.sycl::_V1::accessor.2" addrspace(4)* %{{.*}}, %"class.sycl::_V1::id.2" %{{.*}})

SYCL_EXTERNAL void accessor_subscript_operator_0(sycl::accessor<sycl::cl_int, 2> acc, sycl::id<2> index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_1N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(%{{.*}}: !sycl_accessor_2_i32_read_write_global_buffer, %{{.*}}: i64) attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] {BaseType = memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEEDam, TypeName = @accessor} : (memref<?x!sycl_accessor_2_i32_read_write_global_buffer, 4>, i64) -> !llvm.struct<(!sycl_id_2_, !sycl_accessor_2_i32_read_write_global_buffer)>

// CHECK-LLVM-LABEL: define spir_func void @_Z29accessor_subscript_operator_1N4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(%"class.sycl::_V1::accessor.2" %{{.*}}, i64 %{{.*}}) #0 {
// CHECK-LLVM: %{{.*}} = call { %"class.sycl::_V1::id.2", %"class.sycl::_V1::accessor.2" } @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEEDam(%"class.sycl::_V1::accessor.2" addrspace(4)* %{{.*}}, i64 %1)

SYCL_EXTERNAL void accessor_subscript_operator_1(sycl::accessor<sycl::cl_int, 2> acc, size_t index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z29accessor_subscript_operator_2N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(%{{.*}}: !sycl_accessor_1_i32_read_write_global_buffer, %{{.*}}: i64) attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR: %{{.*}} = sycl.accessor.subscript %{{.*}}[%{{.*}}] {BaseType = memref<?x!sycl_accessor_1_i32_read_write_global_buffer, 4>, FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_1_i32_read_write_global_buffer, 4>, !sycl_id_1_) -> memref<?xi32, 4>

// CHECK-LLVM-LABEL: define spir_func void @_Z29accessor_subscript_operator_2N4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEEm(%"class.sycl::_V1::accessor.1" %0, i64 %1) #0 {
// CHECK-LLVM:  %{{.*}} = call i32 addrspace(4)* @_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)* %{{.*}}, %"class.sycl::_V1::id.1" %{{.*}})

SYCL_EXTERNAL void accessor_subscript_operator_2(sycl::accessor<sycl::cl_int, 1> acc, size_t index) {
  keep(acc[index]);
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_0N4sycl3_V15rangeILi2EEEi(%{{.*}}: !sycl_range_2_, %{{.*}}: i32) attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR: %{{.*}} = "sycl.range.get"(%{{.*}}, %{{.*}}) {BaseType = memref<?x!sycl_array_2_, 4>, FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, TypeName = @array} : (memref<?x!sycl_range_2_, 4>, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z11range_get_0N4sycl3_V15rangeILi2EEEi(%"class.sycl::_V1::range.2" %0, i32 %1) #0 {
// CHECK-LLVM: %{{.*}} = call i64 @_ZNK4sycl3_V16detail5arrayILi2EE3getEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void range_get_0(sycl::range<2> r, int dimension) {
  keep(r.get(dimension));
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_1N4sycl3_V15rangeILi2EEEi(%{{.*}}: !sycl_range_2_, %{{.*}}: i32) attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR: %{{.*}} = "sycl.range.get"(%{{.*}}, %{{.*}}) {BaseType = memref<?x!sycl_array_2_, 4>, FunctionName = @"operator[]", MangledFunctionName = @_ZN4sycl3_V16detail5arrayILi2EEixEi, TypeName = @array} : (memref<?x!sycl_range_2_, 4>, i32) -> memref<?xi64, 4>

// CHECK-LLVM-LABEL: define spir_func void @_Z11range_get_1N4sycl3_V15rangeILi2EEEi(%"class.sycl::_V1::range.2" %0, i32 %1) #0 {
// CHECK-LLVM: %{{.*}} = call i64 addrspace(4)* @_ZN4sycl3_V16detail5arrayILi2EEixEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void range_get_1(sycl::range<2> r, int dimension) {
  keep(r[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z11range_get_2N4sycl3_V15rangeILi2EEEi(%{{.*}}: !sycl_range_2_, %{{.*}}: i32) attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR: %{{.*}} = "sycl.range.get"(%{{.*}}, %{{.*}}) {BaseType = memref<?x!sycl_array_2_, 4>, FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EEixEi, TypeName = @array} : (memref<?x!sycl_range_2_, 4>, i32) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z11range_get_2N4sycl3_V15rangeILi2EEEi(%"class.sycl::_V1::range.2" %0, i32 %1) #0 {
// CHECK-LLVM: %{{.*}} = call i64 @_ZNK4sycl3_V16detail5arrayILi2EEixEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %{{.*}}, i32 %1)

SYCL_EXTERNAL void range_get_2(const sycl::range<2> r, int dimension) {
  keep(r[dimension]);
}

// CHECK-MLIR-LABEL: func.func @_Z10range_sizeN4sycl3_V15rangeILi2EEE(%{{.*}}: !sycl_range_2_) attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR: %{{.*}} = "sycl.range.size"(%{{.*}}) {BaseType = memref<?x!sycl_range_2_, 4>, FunctionName = @size, MangledFunctionName = @_ZNK4sycl3_V15rangeILi2EE4sizeEv, TypeName = @range} : (memref<?x!sycl_range_2_, 4>) -> i64

// CHECK-LLVM-LABEL: define spir_func void @_Z10range_sizeN4sycl3_V15rangeILi2EEE(%"class.sycl::_V1::range.2" %0) #0 {
// CHECK-LLVM: %{{.*}} = call i64 @_ZNK4sycl3_V15rangeILi2EE4sizeEv(%"class.sycl::_V1::range.2" addrspace(4)* %{{.*}})

SYCL_EXTERNAL void range_size(sycl::range<2> r) {
  keep(r.size());
}

// CHECK-MLIR: func.func @_Z8method_1N4sycl3_V14itemILi2ELb1EEE(%arg0: !sycl_item_2_1_)
// CHECK-MLIR-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-MLIR-NEXT: %0 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-MLIR-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_item_2_1_>
// CHECK-MLIR-NEXT: %1 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_item_2_1_>) -> !llvm.ptr<!sycl_item_2_1_>
// CHECK-MLIR-NEXT: %2 = llvm.addrspacecast %1 : !llvm.ptr<!sycl_item_2_1_> to !llvm.ptr<!sycl_item_2_1_, 4>
// CHECK-MLIR-NEXT: %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<!sycl_item_2_1_, 4>) -> memref<?x!sycl_item_2_1_, 4>
// CHECK-MLIR-NEXT: %4 = sycl.call(%3, %c0_i32) {Function = @get_id, MangledName = @_ZNK4sycl3_V14itemILi2ELb1EE6get_idEi, Type = @item} : (memref<?x!sycl_item_2_1_, 4>, i32) -> i64
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

// CHECK-LLVM-LABEL: define spir_func void @_Z8method_1N4sycl3_V14itemILi2ELb1EEE(%"class.sycl::_V1::item.2.true" %0) #0 {
// CHECK-LLVM-NEXT:  %2 = alloca %"class.sycl::_V1::item.2.true", align 8
// CHECK-LLVM-NEXT:  store %"class.sycl::_V1::item.2.true" %0, %"class.sycl::_V1::item.2.true"* %2, align 8
// CHECK-LLVM-NEXT:  %3 = addrspacecast %"class.sycl::_V1::item.2.true"* %2 to %"class.sycl::_V1::item.2.true" addrspace(4)*
// CHECK-LLVM-NEXT:  %4 = call i64 @_ZNK4sycl3_V14itemILi2ELb1EE6get_idEi(%"class.sycl::_V1::item.2.true" addrspace(4)* %3, i32 0)
// CHECK-LLVM-NEXT:  ret void
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void method_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}

// CHECK-MLIR: func.func @_Z8method_2N4sycl3_V14itemILi2ELb1EEE(%arg0: !sycl_item_2_1_)
// CHECK-MLIR-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR-NEXT: %0 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-MLIR-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_item_2_1_>
// CHECK-MLIR-NEXT: %1 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_item_2_1_>) -> !llvm.ptr<!sycl_item_2_1_>
// CHECK-MLIR-NEXT: %2 = llvm.addrspacecast %1 : !llvm.ptr<!sycl_item_2_1_> to !llvm.ptr<!sycl_item_2_1_, 4>
// CHECK-MLIR-NEXT: %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<!sycl_item_2_1_, 4>) -> memref<?x!sycl_item_2_1_, 4>
// CHECK-MLIR-NEXT: %4 = sycl.call(%3, %3) {Function = @"operator==", MangledName = @_ZNK4sycl3_V14itemILi2ELb1EEeqERKS2_, Type = @item} : (memref<?x!sycl_item_2_1_, 4>, memref<?x!sycl_item_2_1_, 4>) -> i8
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

// CHECK-LLVM-LABEL: define spir_func void @_Z8method_2N4sycl3_V14itemILi2ELb1EEE(%"class.sycl::_V1::item.2.true" %0) #0 {
// CHECK-LLVM-NEXT:  %2 = alloca %"class.sycl::_V1::item.2.true", align 8
// CHECK-LLVM-NEXT:  store %"class.sycl::_V1::item.2.true" %0, %"class.sycl::_V1::item.2.true"* %2, align 8
// CHECK-LLVM-NEXT:  %3 = addrspacecast %"class.sycl::_V1::item.2.true"* %2 to %"class.sycl::_V1::item.2.true" addrspace(4)*
// CHECK-LLVM-NEXT:  %4 = call i8 @_ZNK4sycl3_V14itemILi2ELb1EEeqERKS2_(%"class.sycl::_V1::item.2.true" addrspace(4)* %3, %"class.sycl::_V1::item.2.true" addrspace(4)* %3)
// CHECK-LLVM-NEXT:  ret void
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void method_2(sycl::item<2, true> item) {
  auto id = item.operator==(item);
}

// CHECK-MLIR: func.func @_Z21nd_item_get_global_idN4sycl3_V17nd_itemILi2EEE(%arg0: !sycl_nd_item_2_)
// CHECK-MLIR-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR-NEXT: %0 = memref.alloca() : memref<1x!sycl_nd_item_2_>
// CHECK-MLIR-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_nd_item_2_>
// CHECK-MLIR-NEXT: %1 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_nd_item_2_>) -> !llvm.ptr<!sycl_nd_item_2_>
// CHECK-MLIR-NEXT: %2 = llvm.addrspacecast %1 : !llvm.ptr<!sycl_nd_item_2_> to !llvm.ptr<!sycl_nd_item_2_, 4>
// CHECK-MLIR-NEXT: %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<!sycl_nd_item_2_, 4>) -> memref<?x!sycl_nd_item_2_, 4>
// CHECK-MLIR-NEXT: %4 = sycl.call(%3) {Function = @get_global_id, MangledName = @_ZNK4sycl3_V17nd_itemILi2EE13get_global_idEv, Type = @nd_item} : (memref<?x!sycl_nd_item_2_, 4>) -> !sycl_id_2_
// CHECK-MLIR-NEXT: return

// CHECK-LLVM-LABEL: define spir_func void @_Z21nd_item_get_global_idN4sycl3_V17nd_itemILi2EEE(%"class.sycl::_V1::nd_item.2" %0) #0 {
// CHECK-LLVM-NEXT:  %2 = alloca %"class.sycl::_V1::nd_item.2", align 8
// CHECK-LLVM-NEXT:  store %"class.sycl::_V1::nd_item.2" %0, %"class.sycl::_V1::nd_item.2"* %2, align 8
// CHECK-LLVM-NEXT:  %3 = addrspacecast %"class.sycl::_V1::nd_item.2"* %2 to %"class.sycl::_V1::nd_item.2" addrspace(4)*
// CHECK-LLVM-NEXT:  %4 = call %"class.sycl::_V1::id.2" @_ZNK4sycl3_V17nd_itemILi2EE13get_global_idEv(%"class.sycl::_V1::nd_item.2" addrspace(4)* %3)
// CHECK-LLVM-NEXT:  ret void
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void nd_item_get_global_id(sycl::nd_item<2> ndItem) {
  auto id = ndItem.get_global_id();
}

// CHECK-MLIR: func.func @_Z4op_1N4sycl3_V12idILi2EEES2_(%arg0: !sycl_id_2_, %arg1: !sycl_id_2_)
// CHECK-MLIR-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-MLIR-NEXT: %1 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-MLIR-NEXT: affine.store %arg0, %1[0] : memref<1x!sycl_id_2_>
// CHECK-MLIR-NEXT: affine.store %arg1, %0[0] : memref<1x!sycl_id_2_>
// CHECK-MLIR-NEXT: %2 = "polygeist.memref2pointer"(%1) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-MLIR-NEXT: %3 = llvm.addrspacecast %2 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %5 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-MLIR-NEXT: %6 = llvm.addrspacecast %5 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-MLIR-NEXT: %8 = sycl.call(%4, %7) {Function = @"operator==", MangledName = @_ZNK4sycl3_V12idILi2EEeqERKS2_, Type = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_id_2_, 4>) -> i8
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

// CHECK-LLVM-LABEL: define spir_func void @_Z4op_1N4sycl3_V12idILi2EEES2_(%"class.sycl::_V1::id.2" %0, %"class.sycl::_V1::id.2" %1) #0 {
// CHECK-LLVM-NEXT: %3 = alloca %"class.sycl::_V1::id.2", align 8
// CHECK-LLVM-NEXT: %4 = alloca %"class.sycl::_V1::id.2", align 8
// CHECK-LLVM-NEXT: store %"class.sycl::_V1::id.2" %0, %"class.sycl::_V1::id.2"* %4, align 8
// CHECK-LLVM-NEXT: store %"class.sycl::_V1::id.2" %1, %"class.sycl::_V1::id.2"* %3, align 8
// CHECK-LLVM-NEXT: %5 = addrspacecast %"class.sycl::_V1::id.2"* %4 to %"class.sycl::_V1::id.2" addrspace(4)*
// CHECK-LLVM-NEXT: %6 = addrspacecast %"class.sycl::_V1::id.2"* %3 to %"class.sycl::_V1::id.2" addrspace(4)*
// CHECK-LLVM-NEXT: %7 = call i8 @_ZNK4sycl3_V12idILi2EEeqERKS2_(%"class.sycl::_V1::id.2" addrspace(4)* %5, %"class.sycl::_V1::id.2" addrspace(4)* %6)
// CHECK-LLVM-NEXT: ret void
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void op_1(sycl::id<2> a, sycl::id<2> b) {
  auto id = a == b;
}

// CHECK-MLIR: func.func @_Z8static_1N4sycl3_V12idILi2EEES2_(%arg0: !sycl_id_2_, %arg1: !sycl_id_2_)
// CHECK-MLIR-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-MLIR-NEXT: %c1_i32 = arith.constant 1 : i32
// CHECK-MLIR-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-MLIR-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-MLIR-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-MLIR-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_id_2_>
// CHECK-MLIR-NEXT: %2 = sycl.cast(%1) : (memref<?x!sycl_id_2_>) -> memref<?x!sycl_array_2_>
// CHECK-MLIR-NEXT: %3 = "polygeist.memref2pointer"(%2) : (memref<?x!sycl_array_2_>) -> !llvm.ptr<!sycl_array_2_>
// CHECK-MLIR-NEXT: %4 = llvm.addrspacecast %3 : !llvm.ptr<!sycl_array_2_> to !llvm.ptr<!sycl_array_2_, 4>
// CHECK-MLIR-NEXT: %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr<!sycl_array_2_, 4>) -> memref<?x!sycl_array_2_, 4>
// CHECK-MLIR-NEXT: %6 = sycl.call(%5, %c0_i32) {Function = @get, MangledName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, Type = @array} : (memref<?x!sycl_array_2_, 4>, i32) -> i64
// CHECK-MLIR-NEXT: %7 = sycl.call(%5, %c1_i32) {Function = @get, MangledName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, Type = @array} : (memref<?x!sycl_array_2_, 4>, i32) -> i64
// CHECK-MLIR-NEXT: %8 = arith.addi %6, %7 : i64
// CHECK-MLIR-NEXT: %9 = sycl.call(%8) {Function = @abs, MangledName = @_ZN4sycl3_V13absImEENSt9enable_ifIXsr6detail14is_ugenintegerIT_EE5valueES3_E4typeES3_} : (i64) -> i64
// CHECK-MLIR-NEXT: return
// CHECK-MLIR-NEXT: }

// CHECK-LLVM-LABEL: define spir_func void @_Z8static_1N4sycl3_V12idILi2EEES2_(%"class.sycl::_V1::id.2" %0, %"class.sycl::_V1::id.2" %1) #0 {
// CHECK-LLVM-NEXT: %3 = alloca %"class.sycl::_V1::id.2", align 8
// CHECK-LLVM-NEXT: store %"class.sycl::_V1::id.2" %0, %"class.sycl::_V1::id.2"* %3, align 8
// CHECK-LLVM-NEXT: %4 = bitcast %"class.sycl::_V1::id.2"* %3 to %"class.sycl::_V1::detail::array.2"*
// CHECK-LLVM-NEXT: %5 = addrspacecast %"class.sycl::_V1::detail::array.2"* %4 to %"class.sycl::_V1::detail::array.2" addrspace(4)*
// CHECK-LLVM-NEXT: %6 = call i64 @_ZNK4sycl3_V16detail5arrayILi2EE3getEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %5, i32 0)
// CHECK-LLVM-NEXT: %7 = call i64 @_ZNK4sycl3_V16detail5arrayILi2EE3getEi(%"class.sycl::_V1::detail::array.2" addrspace(4)* %5, i32 1)
// CHECK-LLVM-NEXT: %8 = add i64 %6, %7
// CHECK-LLVM-NEXT: %9 = call i64 @_ZN4sycl3_V13absImEENSt9enable_ifIXsr6detail14is_ugenintegerIT_EE5valueES3_E4typeES3_(i64 %8)
// CHECK-LLVM-NEXT: ret void
// CHECK-LLVM-NEXT: }

SYCL_EXTERNAL void static_1(sycl::id<2> a, sycl::id<2> b) {
  auto abs = sycl::abs(a.get(0) + a.get(1));
}

// CHECK-LLVM: attributes #0 = { convergent mustprogress norecurse nounwind }
