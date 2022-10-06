// Copyright (C) Codeplay Software Limited

//===--- functions.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: clang++ -fsycl -fsycl-device-only -emit-mlir %s | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
// CHECK: !sycl_id_2_ = !sycl.id<2>
// CHECK: !sycl_item_2_1_ = !sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>

// CHECK: func.func @_Z8method_1N4sycl3_V14itemILi2ELb1EEE(%arg0: !sycl_item_2_1_)
// CHECK-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %1 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_item_2_1_>) -> !llvm.ptr<!sycl_item_2_1_>
// CHECK-NEXT: %2 = llvm.addrspacecast %1 : !llvm.ptr<!sycl_item_2_1_> to !llvm.ptr<!sycl_item_2_1_, 4>
// CHECK-NEXT: %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<!sycl_item_2_1_, 4>) -> memref<?x!sycl_item_2_1_, 4>
// CHECK-NEXT: %4 = sycl.call(%3, %c0_i32) {Function = @get_id, MangledName = @_ZNK4sycl3_V14itemILi2ELb1EE6get_idEi, Type = @item} : (memref<?x!sycl_item_2_1_, 4>, i32) -> i64
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void method_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}

// CHECK: func.func @_Z8method_2N4sycl3_V14itemILi2ELb1EEE(%arg0: !sycl_item_2_1_)
// CHECK-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %1 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_item_2_1_>) -> !llvm.ptr<!sycl_item_2_1_>
// CHECK-NEXT: %2 = llvm.addrspacecast %1 : !llvm.ptr<!sycl_item_2_1_> to !llvm.ptr<!sycl_item_2_1_, 4>
// CHECK-NEXT: %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<!sycl_item_2_1_, 4>) -> memref<?x!sycl_item_2_1_, 4>
// CHECK-NEXT: %4 = sycl.call(%3, %3) {Function = @"operator==", MangledName = @_ZNK4sycl3_V14itemILi2ELb1EEeqERKS2_, Type = @item} : (memref<?x!sycl_item_2_1_, 4>, memref<?x!sycl_item_2_1_, 4>) -> i8
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void method_2(sycl::item<2, true> item) {
  auto id = item.operator==(item);
}

// CHECK: func.func @_Z4op_1N4sycl3_V12idILi2EEES2_(%arg0: !sycl_id_2_, %arg1: !sycl_id_2_)
// CHECK-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg0, %1[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg1, %0[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: %2 = "polygeist.memref2pointer"(%1) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %3 = llvm.addrspacecast %2 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: %5 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %6 = llvm.addrspacecast %5 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: %8 = sycl.call(%4, %7) {Function = @"operator==", MangledName = @_ZNK4sycl3_V12idILi2EEeqERKS2_, Type = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_id_2_, 4>) -> i8
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void op_1(sycl::id<2> a, sycl::id<2> b) {
  auto id = a == b;
}

// CHECK: func.func @_Z8static_1N4sycl3_V12idILi2EEES2_(%arg0: !sycl_id_2_, %arg1: !sycl_id_2_)
// CHECK-SAME: attributes {llvm.cconv = #llvm.cconv<spir_funccc>, llvm.linkage = #llvm.linkage<external>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: %2 = sycl.cast(%1) : (memref<?x!sycl_id_2_>) -> memref<?x!sycl_array_2_>
// CHECK-NEXT: %3 = "polygeist.memref2pointer"(%2) : (memref<?x!sycl_array_2_>) -> !llvm.ptr<!sycl_array_2_>
// CHECK-NEXT: %4 = llvm.addrspacecast %3 : !llvm.ptr<!sycl_array_2_> to !llvm.ptr<!sycl_array_2_, 4>
// CHECK-NEXT: %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr<!sycl_array_2_, 4>) -> memref<?x!sycl_array_2_, 4>
// CHECK-NEXT: %6 = sycl.call(%5, %c0_i32) {Function = @get, MangledName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, Type = @array} : (memref<?x!sycl_array_2_, 4>, i32) -> i64
// CHECK-NEXT: %7 = sycl.call(%5, %c1_i32) {Function = @get, MangledName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, Type = @array} : (memref<?x!sycl_array_2_, 4>, i32) -> i64
// CHECK-NEXT: %8 = arith.addi %6, %7 : i64
// CHECK-NEXT: %9 = sycl.call(%8) {Function = @abs, MangledName = @_ZN4sycl3_V13absImEENSt9enable_ifIXsr6detail14is_ugenintegerIT_EE5valueES3_E4typeES3_} : (i64) -> i64
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void static_1(sycl::id<2> a, sycl::id<2> b) {
  auto abs = sycl::abs(a.get(0) + a.get(1));
}
