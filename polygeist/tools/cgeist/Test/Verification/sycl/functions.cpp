// Copyright (C) Codeplay Software Limited

//===--- functions.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: sycl-clang.py %s -S | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
// CHECK: !sycl_id_2_ = !sycl.id<2>
// CHECK: !sycl_item_2_1_ = !sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>

// CHECK: func.func @_Z8method_1N2cl4sycl4itemILi2ELb1EEE(%arg0: !sycl_item_2_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_item_2_1_> to memref<?x!sycl_item_2_1_>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %2 = sycl.call(%1, %c0_i32) {Function = @get_id, MangledName = @_ZNK2cl4sycl4itemILi2ELb1EE6get_idEi, Type = @item} : (memref<?x!sycl_item_2_1_>, i32) -> i64
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void method_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}

// CHECK: func.func @_Z8method_2N2cl4sycl4itemILi2ELb1EEE(%arg0: !sycl_item_2_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_item_2_1_> to memref<?x!sycl_item_2_1_>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %2 = sycl.call(%1, %1) {Function = @"operator==", MangledName = @_ZNK2cl4sycl4itemILi2ELb1EEeqERKS2_, Type = @item} : (memref<?x!sycl_item_2_1_>, memref<?x!sycl_item_2_1_>) -> i8
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void method_2(sycl::item<2, true> item) {
  auto id = item.operator==(item);
}

// CHECK: func.func @_Z4op_1N2cl4sycl2idILi2EEES2_(%arg0: !sycl_id_2_, %arg1: !sycl_id_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: %2 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %3 = memref.cast %2 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg0, %2[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg1, %0[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: %4 = sycl.call(%3, %1) {Function = @"operator==", MangledName = @_ZNK2cl4sycl2idILi2EEeqERKS2_, Type = @id} : (memref<?x!sycl_id_2_>, memref<?x!sycl_id_2_>) -> i8
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void op_1(sycl::id<2> a, sycl::id<2> b) {
  auto id = a == b;
}

// CHECK: func.func @_Z8static_1N2cl4sycl2idILi2EEES2_(%arg0: !sycl_id_2_, %arg1: !sycl_id_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg0, %0[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: %2 = sycl.cast(%1) : (memref<?x!sycl_id_2_>) -> memref<?x!sycl_array_2_>
// CHECK-NEXT: %3 = sycl.call(%2, %c0_i32) {Function = @get, MangledName = @_ZNK2cl4sycl6detail5arrayILi2EE3getEi, Type = @array} : (memref<?x!sycl_array_2_>, i32) -> i64
// CHECK-NEXT: %4 = sycl.call(%2, %c1_i32) {Function = @get, MangledName = @_ZNK2cl4sycl6detail5arrayILi2EE3getEi, Type = @array} : (memref<?x!sycl_array_2_>, i32) -> i64
// CHECK-NEXT: %5 = arith.addi %3, %4 : i64
// CHECK-NEXT: %6 = sycl.call(%5) {Function = @abs, MangledName = @_ZN2cl4sycl3absImEENSt9enable_ifIXsr6detail14is_ugenintegerIT_EE5valueES3_E4typeES3_} : (i64) -> i64
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void static_1(sycl::id<2> a, sycl::id<2> b) {
  auto abs = sycl::abs(a.get(0) + a.get(1));
}
