//===--- constructors.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
// RUN: sycl-clang.py %s -S 2> /dev/null | FileCheck %s --check-prefixes=CHECK
// RUN: sycl-clang.py %s -S -emit-llvm | FileCheck %s --check-prefixes=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-DAG: !sycl_id_2_ = !sycl.id<2>
// CHECK-DAG: !sycl_item_2_1_ = !sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>
// CHECK-DAG: !sycl_range_1_ = !sycl.range<1>

// Ensure the constructors are NOT filtered out, and sycl.cast is generated for cast from sycl.id or sycl.range to sycl.array.
// CHECK:      func.func @_ZN4sycl3_V12idILi1EEC1ERKS2_(%arg0: memref<?x!sycl_id_1_>, %arg1: memref<?x!sycl_id_1_>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:   %0 = sycl.cast(%arg0) : (memref<?x!sycl_id_1_>) -> memref<?x!sycl_array_1_>
// CHECK:      func.func @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%arg0: memref<?x!sycl_range_1_>, %arg1: memref<?x!sycl_range_1_>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:   %0 = sycl.cast(%arg0) : (memref<?x!sycl_range_1_>) -> memref<?x!sycl_array_1_>

// CHECK-LLVM: define void @cons_0([[ID_TYPE:%"class.cl::sycl::id.1"]] [[ARG0:%.*]], [[RANGE_TYPE:%"class.cl::sycl::range.1"]] [[ARG1:%.*]]) {
// CHECK-LLVM-DAG: [[RANGE1:%.*]] = alloca [[RANGE_TYPE]]
// CHECK-LLVM-DAG: [[ID1:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM-DAG: [[RANGE2:%.*]] = alloca [[RANGE_TYPE]]
// CHECK-LLVM-DAG: [[ID2:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM: store [[ID_TYPE]] [[ARG0]], [[ID_TYPE]]* [[ID2]]
// CHECK-LLVM: store [[RANGE_TYPE]] [[ARG1]], [[RANGE_TYPE]]* [[RANGE2]]
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi1EEC1ERKS2_([[ID_TYPE]]* [[ID1]], 
// CHECK-LLVM: call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_([[RANGE_TYPE]]* [[RANGE1]],

extern "C" SYCL_EXTERNAL void cons_0(sycl::id<1> i, sycl::range<1> r) {
  auto id = sycl::id<1>{i};
  auto range = sycl::range<1>{r};
}

// CHECK: func.func @cons_1() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %false = arith.constant false
// CHECK-NEXT: %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: %2 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<i8>
// CHECK-NEXT: %3 = "polygeist.typeSize"() {source = !sycl_id_2_} : () -> index
// CHECK-NEXT: %4 = arith.index_cast %3 : index to i64
// CHECK-NEXT: "llvm.intr.memset"(%2, %c0_i8, %4, %false) : (!llvm.ptr<i8>, i8, i64, i1) -> ()
// CHECK-NEXT: sycl.constructor(%1) {MangledName = @_ZN4sycl3_V12idILi2EEC1Ev, Type = @id} : (memref<?x!sycl_id_2_>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

// Ensure declaration to have external linkage.
// CHECK: func.func private @_ZN4sycl3_V12idILi2EEC1Ev(memref<?x!sycl_id_2_>) attributes {llvm.linkage = #llvm.linkage<external>}

// CHECK-LLVM: define void @cons_1() {
// CHECK-LLVM: [[ID1:%.*]] = alloca [[ID_TYPE:%"class.cl::sycl::id.2"]]
// CHECK-LLVM: [[CAST1:%.*]] = bitcast [[ID_TYPE]]* %1 to i8*
// CHECK-LLVM: call void @llvm.memset.p0i8.i64(i8* %2, i8 0, i64 16, i1 false)
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1Ev([[ID_TYPE]]* [[ID1]], [[ID_TYPE]]* [[ID1]], i64 0, i64 1, i64 1)  

extern "C" SYCL_EXTERNAL void cons_1() {
  auto id = sycl::id<2>{};
}

// CHECK: func.func @cons_2(%arg0: i64, %arg1: i64) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: sycl.constructor(%1, %arg0, %arg1) {MangledName = @_ZN4sycl3_V12idILi2EEC1ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm, Type = @id} : (memref<?x!sycl_id_2_>, i64, i64) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }


// CHECK-LLVM: define void @cons_2(i64 %0, i64 %1) {
// CHECK-LLVM: [[ID1:%.*]] = alloca [[ID_TYPE:%"class.cl::sycl::id.2"]]
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[ID_TYPE]]* [[ID1]], [[ID_TYPE]]* [[ID1]], i64 0, i64 1, i64 1, i64 %0, i64 %1)

extern "C" SYCL_EXTERNAL void cons_2(size_t a, size_t b) {
  auto id = sycl::id<2>{a, b};
}
 
// CHECK: func.func @cons_3(%arg0: !sycl_item_2_1_) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: %2 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %3 = memref.cast %2 : memref<1x!sycl_item_2_1_> to memref<?x!sycl_item_2_1_>
// CHECK-NEXT: affine.store %arg0, %2[0] : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: sycl.constructor(%1, %3) {MangledName = @_ZN4sycl3_V12idILi2EEC1ILi2ELb1EEERNSt9enable_ifIXeqT_Li2EEKNS0_4itemILi2EXT0_EEEE4typeE, Type = @id} : (memref<?x!sycl_id_2_>, memref<?x!sycl_item_2_1_>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LLVM: define void @cons_3([[ITEM_TYPE:%"class.cl::sycl::item.2.true"]] [[ARG0:%.*]]) {
// CHECK-LLVM-DAG: [[ID:%.*]] = alloca [[ID_TYPE:%"class.cl::sycl::id.2"]]  
// CHECK-LLVM-DAG: [[ITEM:%.*]] = alloca [[ITEM_TYPE]]
// CHECK-LLVM: store [[ITEM_TYPE]] [[ARG0]], [[ITEM_TYPE]]* [[ITEM]], align 8
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ILi2ELb1EEERNSt9enable_ifIXeqT_Li2EEKNS0_4itemILi2EXT0_EEEE4typeE([[ID_TYPE]]* [[ID]], [[ID_TYPE]]* [[ID]], i64 0, i64 1, i64 1, [[ITEM_TYPE]]* [[ITEM]], [[ITEM_TYPE]]* [[ITEM]], i64 0, i64 1, i64 1)

extern "C" SYCL_EXTERNAL void cons_3(sycl::item<2, true> val) {
  auto id = sycl::id<2>{val};
}

// CHECK: func.func @cons_4(%arg0: !sycl_id_2_) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.cast %0 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: %2 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %3 = memref.cast %2 : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg0, %2[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: sycl.constructor(%1, %3) {MangledName = @_ZN4sycl3_V12idILi2EEC1ERKS2_, Type = @id} : (memref<?x!sycl_id_2_>, memref<?x!sycl_id_2_>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LLVM: define void @cons_4([[ID_TYPE:%"class.cl::sycl::id.2"]] [[ARG0:%.*]]) {
// CHECK-LLVM-DAG: [[ID1:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM-DAG: [[ID2:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM: store [[ID_TYPE]] [[ARG0]], [[ID_TYPE]]* [[ID2]], align 8
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ERKS2_([[ID_TYPE]]* [[ID1]], [[ID_TYPE]]* [[ID1]], i64 0, i64 1, i64 1, [[ID_TYPE]]* [[ID2]], [[ID_TYPE]]* [[ID2]], i64 0, i64 1, i64 1)

extern "C"  SYCL_EXTERNAL void cons_4(sycl::id<2> val) {
  auto id = sycl::id<2>{val};
}
