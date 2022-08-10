// Copyright (C) Codeplay Software Limited

//===--- constructors.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: sycl-clang.py %s -S 2> /dev/null | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: !sycl_id_2_ = !sycl.id<2>
// CHECK: !sycl_item_2_1_ = !sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>

// CHECK: func.func @_Z6cons_1v() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %false = arith.constant false
// CHECK-NEXT: %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: %2 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<i8>
// CHECK-NEXT: %3 = "polygeist.typeSize"() {source = !sycl_id_2_} : () -> index
// CHECK-NEXT: %4 = arith.index_cast %3 : index to i64
// CHECK-NEXT: "llvm.intr.memset"(%2, %c0_i8, %4, %false) : (!llvm.ptr<i8>, i8, i64, i1) -> ()
// CHECK-NEXT: sycl.constructor(%1) {Type = @id} : (memref<?x!sycl_id_2_>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void cons_1() {
  auto id = sycl::id<2>{};
}

// CHECK: func.func @_Z6cons_2mm(%arg0: i64, %arg1: i64) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: sycl.constructor(%1, %arg0, %arg1) {Type = @id} : (memref<?x!sycl_id_2_>, i64, i64) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void cons_2(size_t a, size_t b) {
  auto id = sycl::id<2>{a, b};
}

// CHECK: func.func @_Z6cons_3N2cl4sycl4itemILi2ELb1EEE(%arg0: !sycl_item_2_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: %2 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %3 = memref.cast %2 : memref<1x!sycl_item_2_1_> to memref<?x!sycl_item_2_1_>
// CHECK-NEXT: affine.store %arg0, %2[0] : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: sycl.constructor(%1, %3) {Type = @id} : (memref<?x!sycl_id_2_>, memref<?x!sycl_item_2_1_>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void cons_3(sycl::item<2, true> val) {
  auto id = sycl::id<2>{val};
}

// CHECK: func.func @_Z6cons_4N2cl4sycl2idILi2EEE(%arg0: !sycl_id_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: %2 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %3 = memref.cast %2 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg0, %2[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: sycl.constructor(%1, %3) {Type = @id} : (memref<?x!sycl_id_2_>, memref<?x!sycl_id_2_>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

SYCL_EXTERNAL void cons_4(sycl::id<2> val) {
  auto id = sycl::id<2>{val};
}
