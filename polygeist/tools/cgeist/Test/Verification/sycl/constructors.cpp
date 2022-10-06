//===--- constructors.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
// RUN: clang++ -fsycl -fsycl-device-only -emit-mlir %s 2> /dev/null | FileCheck %s --check-prefixes=CHECK
// RUN: clang++ -fsycl -fsycl-device-only -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s | FileCheck %s --check-prefixes=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-DAG: !sycl_id_2_ = !sycl.id<2>
// CHECK-DAG: !sycl_item_2_1_ = !sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>
// CHECK-DAG: !sycl_range_1_ = !sycl.range<1>

// Ensure the constructors are NOT filtered out, and sycl.cast is generated for cast from sycl.id or sycl.range to sycl.array.
// CHECK-LABEL: func.func @_ZN4sycl3_V12idILi1EEC1ERKS2_(%arg0: memref<?x!sycl_id_1_, 4>, %arg1: memref<?x!sycl_id_1_, 4>)
// CHECK-SAME:  attributes {[[SPIR_FUNCCC:llvm.cconv = #llvm.cconv<spir_funccc>]], [[LINKONCE:llvm.linkage = #llvm.linkage<linkonce_odr>]], 
// CHECK-SAME:  [[PASSTHROUGH:passthrough = \["norecurse", "nounwind", "convergent", "mustprogress"\]]]} {
// CHECK-NEXT:   %0 = sycl.cast(%arg0) : (memref<?x!sycl_id_1_, 4>) -> memref<?x!sycl_array_1_, 4>
// CHECK-LABEL: func.func @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%arg0: memref<?x!sycl_range_1_, 4>, %arg1: memref<?x!sycl_range_1_, 4>)
// CHECK-SAME:  attributes {[[SPIR_FUNCCC]], [[LINKONCE]], [[PASSTHROUGH]]}
// CHECK-NEXT:   %0 = sycl.cast(%arg0) : (memref<?x!sycl_range_1_, 4>) -> memref<?x!sycl_array_1_, 4>

// CHECK-LLVM: define spir_func void @cons_0([[ID_TYPE:%"class.sycl::_V1::id.1"]] [[ARG0:%.*]], [[RANGE_TYPE:%"class.sycl::_V1::range.1"]] [[ARG1:%.*]]) #0
// CHECK-LLVM-DAG: [[RANGE1:%.*]] = alloca [[RANGE_TYPE]]
// CHECK-LLVM-DAG: [[ID1:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM-DAG: [[RANGE2:%.*]] = alloca [[RANGE_TYPE]]
// CHECK-LLVM-DAG: [[ID2:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM: store [[ID_TYPE]] [[ARG0]], [[ID_TYPE]]* [[ID2]]
// CHECK-LLVM: store [[RANGE_TYPE]] [[ARG1]], [[RANGE_TYPE]]* [[RANGE2]]
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID1]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi1EEC1ERKS2_([[ID_TYPE]] addrspace(4)* [[ID1_AS]], 
// CHECK-LLVM: [[RANGE1_AS:%.*]] = addrspacecast [[RANGE_TYPE]]* [[RANGE1]] to [[RANGE_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_([[RANGE_TYPE]] addrspace(4)* [[RANGE1_AS]],

extern "C" SYCL_EXTERNAL void cons_0(sycl::id<1> i, sycl::range<1> r) {
  auto id = sycl::id<1>{i};
  auto range = sycl::range<1>{r};
}

// CHECK-LABEL: func.func @cons_1()
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL:llvm.linkage = #llvm.linkage<external>]], [[PASSTHROUGH]]} {
// CHECK-NEXT: %false = arith.constant false
// CHECK-NEXT: %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<i8>
// CHECK-NEXT: %2 = "polygeist.typeSize"() {source = !sycl_id_2_} : () -> index
// CHECK-NEXT: %3 = arith.index_cast %2 : index to i64
// CHECK-NEXT: "llvm.intr.memset"(%1, %c0_i8, %3, %false) : (!llvm.ptr<i8>, i8, i64, i1) -> ()
// CHECK-NEXT: %4 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %5 = llvm.addrspacecast %4 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %6 = "polygeist.pointer2memref"(%5) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: sycl.constructor(%6) {MangledName = @_ZN4sycl3_V12idILi2EEC1Ev, Type = @id} : (memref<?x!sycl_id_2_, 4>) -> ()

// Ensure declaration to have external linkage.
// CHECK-LABEL: func.func private @_ZN4sycl3_V12idILi2EEC1Ev(memref<?x!sycl_id_2_, 4>)
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], [[PASSTHROUGH]]}

// CHECK-LLVM-LABEL: define spir_func void @cons_1() #0
// CHECK-LLVM: [[ID1:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]
// CHECK-LLVM: [[CAST1:%.*]] = bitcast [[ID_TYPE]]* %1 to i8*
// CHECK-LLVM: call void @llvm.memset.p0i8.i64(i8* %2, i8 0, i64 16, i1 false)
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID1]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1Ev([[ID_TYPE]] addrspace(4)* [[ID1_AS]], [[ID_TYPE]] addrspace(4)* [[ID1_AS]], i64 0, i64 -1, i64 1)  

extern "C" SYCL_EXTERNAL void cons_1() {
  auto id = sycl::id<2>{};
}

// CHECK-LABEL: func.func @cons_2(%arg0: i64, %arg1: i64)
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], [[PASSTHROUGH]]}
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %2 = llvm.addrspacecast %1 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %3 = "polygeist.pointer2memref"(%2) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: sycl.constructor(%3, %arg0, %arg1) {MangledName = @_ZN4sycl3_V12idILi2EEC1ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm, Type = @id} : (memref<?x!sycl_id_2_, 4>, i64, i64) -> ()

// CHECK-LLVM-LABEL: define spir_func void @cons_2(i64 %0, i64 %1) #0
// CHECK-LLVM: [[ID1:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID1]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[ID_TYPE]] addrspace(4)* [[ID1_AS]], [[ID_TYPE]] addrspace(4)* [[ID1_AS]], i64 0, i64 -1, i64 1, i64 %0, i64 %1)

extern "C" SYCL_EXTERNAL void cons_2(size_t a, size_t b) {
  auto id = sycl::id<2>{a, b};
}

// CHECK-LABEL: func.func @cons_3(%arg0: !sycl_item_2_1_)
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], [[PASSTHROUGH]]}
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.alloca() : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: affine.store %arg0, %1[0] : memref<1x!sycl_item_2_1_>
// CHECK-NEXT: %2 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %3 = llvm.addrspacecast %2 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: %5 = "polygeist.memref2pointer"(%1) : (memref<1x!sycl_item_2_1_>) -> !llvm.ptr<!sycl_item_2_1_>
// CHECK-NEXT: %6 = llvm.addrspacecast %5 : !llvm.ptr<!sycl_item_2_1_> to !llvm.ptr<!sycl_item_2_1_, 4>
// CHECK-NEXT: %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr<!sycl_item_2_1_, 4>) -> memref<?x!sycl_item_2_1_, 4>
// CHECK-NEXT: sycl.constructor(%4, %7) {MangledName = @_ZN4sycl3_V12idILi2EEC1ILi2ELb1EEERNSt9enable_ifIXeqT_Li2EEKNS0_4itemILi2EXT0_EEEE4typeE, Type = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_item_2_1_, 4>) -> ()

// CHECK-LLVM: define spir_func void @cons_3([[ITEM_TYPE:%"class.sycl::_V1::item.2.true"]] [[ARG0:%.*]]) #0
// CHECK-LLVM-DAG: [[ID:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]  
// CHECK-LLVM-DAG: [[ITEM:%.*]] = alloca [[ITEM_TYPE]]
// CHECK-LLVM: store [[ITEM_TYPE]] [[ARG0]], [[ITEM_TYPE]]* [[ITEM]], align 8
// CHECK-LLVM: [[ID_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: [[ITEM_AS:%.*]] = addrspacecast [[ITEM_TYPE]]* [[ITEM]] to [[ITEM_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ILi2ELb1EEERNSt9enable_ifIXeqT_Li2EEKNS0_4itemILi2EXT0_EEEE4typeE([[ID_TYPE]] addrspace(4)* [[ID_AS]], [[ID_TYPE]] addrspace(4)* [[ID_AS]], i64 0, i64 -1, i64 1, [[ITEM_TYPE]] addrspace(4)* [[ITEM_AS]], [[ITEM_TYPE]] addrspace(4)* [[ITEM_AS]], i64 0, i64 -1, i64 1)

extern "C" SYCL_EXTERNAL void cons_3(sycl::item<2, true> val) {
  auto id = sycl::id<2>{val};
}

// CHECK-LABEL: func.func @cons_4(%arg0: !sycl_id_2_)
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], [[PASSTHROUGH]]}
// CHECK-NEXT: %0 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %1 = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: affine.store %arg0, %1[0] : memref<1x!sycl_id_2_>
// CHECK-NEXT: %2 = "polygeist.memref2pointer"(%0) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %3 = llvm.addrspacecast %2 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: %5 = "polygeist.memref2pointer"(%1) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %6 = llvm.addrspacecast %5 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %7 = "polygeist.pointer2memref"(%6) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: sycl.constructor(%4, %7) {MangledName = @_ZN4sycl3_V12idILi2EEC1ERKS2_, Type = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_id_2_, 4>) -> ()

// CHECK-LLVM: define spir_func void @cons_4([[ID_TYPE:%"class.sycl::_V1::id.2"]] [[ARG0:%.*]]) #0
// CHECK-LLVM-DAG: [[ID1:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM-DAG: [[ID2:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM: store [[ID_TYPE]] [[ARG0]], [[ID_TYPE]]* [[ID2]], align 8
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID1]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: [[ID2_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID2]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ERKS2_([[ID_TYPE]] addrspace(4)* [[ID1_AS]], [[ID_TYPE]] addrspace(4)* [[ID1_AS]], i64 0, i64 -1, i64 1, [[ID_TYPE]] addrspace(4)* [[ID2_AS]], [[ID_TYPE]] addrspace(4)* [[ID2_AS]], i64 0, i64 -1, i64 1)

extern "C" SYCL_EXTERNAL void cons_4(sycl::id<2> val) {
  auto id = sycl::id<2>{val};
}

// CHECK-LABEL: func.func @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev({{.*}})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKONCE]], [[PASSTHROUGH]]}
// CHECK: [[I:%.*]] = "polygeist.subindex"(%arg0, %c0) : (memref<?x!sycl_accessor_1_i32_write_global_buffer, 4>, index) -> memref<?x!sycl_accessor_impl_device_1_, 4>
// CHECK: sycl.constructor([[I]], {{%.*}}, {{%.*}}, {{%.*}}) {MangledName = @_ZN4sycl3_V16detail18AccessorImplDeviceILi1EEC1ENS0_2idILi1EEENS0_5rangeILi1EEES7_, Type = @AccessorImplDevice} : (memref<?x!sycl_accessor_impl_device_1_, 4>, !sycl_id_1_, !sycl_range_1_, !sycl_range_1_) -> ()

// CHECK-LLVM-LABEL: define spir_func void @cons_5() #0
// CHECK-LLVM: [[ACCESSOR:%.*]] = alloca %"class.sycl::_V1::accessor.1", align 8
// CHECK-LLVM: [[ACAST:%.*]] = addrspacecast %"class.sycl::_V1::accessor.1"* [[ACCESSOR]] to %"class.sycl::_V1::accessor.1" addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* [[ACAST]], %"class.sycl::_V1::accessor.1" addrspace(4)* [[ACAST]], i64 0, i64 -1, i64 1)

extern "C" SYCL_EXTERNAL void cons_5() {
  auto accessor = sycl::accessor<sycl::cl_int, 1, sycl::access::mode::write>{};
}

// Keep at the end.
// CHECK-LLVM: attributes #0 = { convergent mustprogress norecurse nounwind }
