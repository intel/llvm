// RUN: clang++ -fsycl -fsycl-device-only -emit-mlir %s -o - 2> /dev/null | FileCheck %s --check-prefixes=CHECK
// RUN: clang++ -fsycl -fsycl-device-only -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefixes=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-DAG: !sycl_id_2_ = !sycl.id<2>
// CHECK-DAG: !sycl_item_2_1_ = !sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl.range<2>, !sycl.id<2>, !sycl.id<2>)>)>
// CHECK-DAG: !sycl_range_1_ = !sycl.range<1>

// Check globals referenced in device functions are created in the GPU module
// CHECK: gpu.module @device_functions {
// CHECK-DAG: memref.global constant @__spirv_BuiltInSubgroupLocalInvocationId : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInSubgroupId : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInNumSubgroups : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInSubgroupMaxSize : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInSubgroupSize : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInLocalInvocationId : memref<3xi64, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInWorkgroupId : memref<3xi64, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInWorkgroupSize : memref<3xi64, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInNumWorkgroups : memref<3xi64, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInGlobalOffset : memref<3xi64, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInGlobalSize : memref<3xi64, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInGlobalInvocationId : memref<3xi64, 1> {alignment = 32 : i64}

// Ensure the spirv functions that reference these globals are not filtered out
// CHECK-DAG: func.func @_Z28__spirv_GlobalInvocationId_xv()
// CHECK-DAG: func.func @_Z28__spirv_GlobalInvocationId_yv()
// CHECK-DAG: func.func @_Z28__spirv_GlobalInvocationId_zv()
// CHECK-DAG: func.func @_Z20__spirv_GlobalSize_xv()
// CHECK-DAG: func.func @_Z20__spirv_GlobalSize_yv()
// CHECK-DAG: func.func @_Z20__spirv_GlobalSize_zv()
// CHECK-DAG: func.func @_Z22__spirv_GlobalOffset_xv()
// CHECK-DAG: func.func @_Z22__spirv_GlobalOffset_yv()
// CHECK-DAG: func.func @_Z22__spirv_GlobalOffset_zv()
// CHECK-DAG: func.func @_Z23__spirv_NumWorkgroups_xv()
// CHECK-DAG: func.func @_Z23__spirv_NumWorkgroups_yv()
// CHECK-DAG: func.func @_Z23__spirv_NumWorkgroups_zv()
// CHECK-DAG: func.func @_Z23__spirv_WorkgroupSize_xv()
// CHECK-DAG: func.func @_Z23__spirv_WorkgroupSize_yv()
// CHECK-DAG: func.func @_Z23__spirv_WorkgroupSize_zv()
// CHECK-DAG: func.func @_Z21__spirv_WorkgroupId_xv()
// CHECK-DAG: func.func @_Z21__spirv_WorkgroupId_yv()
// CHECK-DAG: func.func @_Z21__spirv_WorkgroupId_zv()
// CHECK-DAG: func.func @_Z27__spirv_LocalInvocationId_xv()
// CHECK-DAG: func.func @_Z27__spirv_LocalInvocationId_yv()
// CHECK-DAG: func.func @_Z27__spirv_LocalInvocationId_zv()
// CHECK-DAG: func.func @_Z18__spirv_SubgroupIdv()
// CHECK-DAG: func.func @_Z33__spirv_SubgroupLocalInvocationIdv()
// CHECK-DAG: func.func @_Z23__spirv_SubgroupMaxSizev()
// CHECK-DAG: func.func @_Z20__spirv_NumSubgroupsv()

// Ensure the constructors are NOT filtered out, and sycl.cast is generated for cast from sycl.id or sycl.range to sycl.array.
// CHECK-LABEL: func.func @_ZN4sycl3_V12idILi1EEC1ERKS2_
// CHECK:          (%arg0: memref<?x!sycl_id_1_, 4> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 8 : i64, llvm.noundef}, 
// CHECK-SAME:      %arg1: memref<?x!sycl_id_1_, 4> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.noundef})
// CHECK-SAME:      attributes {[[SPIR_FUNCCC:llvm.cconv = #llvm.cconv<spir_funccc>]], [[LINKONCE:llvm.linkage = #llvm.linkage<linkonce_odr>]] 
// CHECK-NEXT:   %0 = sycl.cast(%arg0) : (memref<?x!sycl_id_1_, 4>) -> memref<?x!sycl_array_1_, 4>
//
// CHECK-LABEL: func.func @_ZN4sycl3_V15rangeILi1EEC1ERKS2_
// CHECK:          (%arg0: memref<?x!sycl_range_1_, 4> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 8 : i64, llvm.noundef}, 
// CHECK-SAME:      %arg1: memref<?x!sycl_range_1_, 4> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.noundef})
// CHECK-SAME:     attributes {[[SPIR_FUNCCC]], [[LINKONCE]], {{.*}}}
// CHECK-NEXT:   %0 = sycl.cast(%arg0) : (memref<?x!sycl_range_1_, 4>) -> memref<?x!sycl_array_1_, 4>

// CHECK-LLVM: define spir_func void @cons_0([[ID_TYPE:%"class.sycl::_V1::id.1"]]* noundef byval(%"class.sycl::_V1::id.1") align 8 [[ARG0:%.*]], 
// CHECK-LLVM-SAME:                          [[RANGE_TYPE:%"class.sycl::_V1::range.1"]]* noundef byval(%"class.sycl::_V1::range.1") align 8 [[ARG1:%.*]]) #1
// CHECK-LLVM-DAG: [[RANGE:%.*]] = alloca [[RANGE_TYPE]]
// CHECK-LLVM-DAG: [[ID:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi1EEC1ERKS2_([[ID_TYPE]] addrspace(4)* [[ID1_AS]], 
// CHECK-LLVM: [[RANGE1_AS:%.*]] = addrspacecast [[RANGE_TYPE]]* [[RANGE]] to [[RANGE_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_([[RANGE_TYPE]] addrspace(4)* [[RANGE1_AS]],

extern "C" SYCL_EXTERNAL void cons_0(sycl::id<1> i, sycl::range<1> r) {
  auto id = sycl::id<1>{i};
  auto range = sycl::range<1>{r};
}

// CHECK-LABEL: func.func @cons_1()
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL:llvm.linkage = #llvm.linkage<external>]], {{.*}}} {
// CHECK-NEXT: %false = arith.constant false
// CHECK-NEXT: %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT: %alloca = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<i8>
// CHECK-NEXT: %1 = "polygeist.typeSize"() {source = !sycl_id_2_} : () -> index
// CHECK-NEXT: %2 = arith.index_cast %1 : index to i64
// CHECK-NEXT: "llvm.intr.memset"(%0, %c0_i8, %2, %false) : (!llvm.ptr<i8>, i8, i64, i1) -> ()
// CHECK-NEXT: %3 = "polygeist.memref2pointer"(%alloca) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %4 = llvm.addrspacecast %3 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: sycl.constructor(%5) {MangledName = @_ZN4sycl3_V12idILi2EEC1Ev, Type = @id} : (memref<?x!sycl_id_2_, 4>) -> ()

// Ensure declaration to have external linkage.
// CHECK-LABEL: func.func private @_ZN4sycl3_V12idILi2EEC1Ev(memref<?x!sycl_id_2_, 4> {llvm.align = 8 : i64, llvm.dereferenceable_or_null = 16 : i64, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], {{.*}}}

// CHECK-LLVM-LABEL: define spir_func void @cons_1() #1
// CHECK-LLVM: [[ID1:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]
// CHECK-LLVM: [[CAST1:%.*]] = bitcast [[ID_TYPE]]* %1 to i8*
// CHECK-LLVM: call void @llvm.memset.p0i8.i64(i8* %2, i8 0, i64 16, i1 false)
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID1]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1Ev([[ID_TYPE]] addrspace(4)* [[ID1_AS]])  

extern "C" SYCL_EXTERNAL void cons_1() {
  auto id = sycl::id<2>{};
}

// CHECK-LABEL: func.func @cons_2(%arg0: i64 {llvm.noundef}, %arg1: i64 {llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], {{.*}}}
// CHECK-NEXT: %alloca = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %1 = llvm.addrspacecast %0 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: sycl.constructor(%2, %arg0, %arg1) {MangledName = @_ZN4sycl3_V12idILi2EEC1ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm, Type = @id} : (memref<?x!sycl_id_2_, 4>, i64, i64) -> ()

// CHECK-LLVM-LABEL: define spir_func void @cons_2(i64 noundef %0, i64 noundef %1) #1
// CHECK-LLVM: [[ID1:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID1]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[ID_TYPE]] addrspace(4)* [[ID1_AS]], i64 %0, i64 %1)

extern "C" SYCL_EXTERNAL void cons_2(size_t a, size_t b) {
  auto id = sycl::id<2>{a, b};
}

// CHECK-LABEL: func.func @cons_3(%arg0: memref<?x!sycl_item_2_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_item_2_1_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], {{.*}}}
// CHECK-NEXT: %alloca = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %1 = llvm.addrspacecast %0 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: %3 = "polygeist.memref2pointer"(%arg0) : (memref<?x!sycl_item_2_1_>) -> !llvm.ptr<!sycl_item_2_1_>
// CHECK-NEXT: %4 = llvm.addrspacecast %3 : !llvm.ptr<!sycl_item_2_1_> to !llvm.ptr<!sycl_item_2_1_, 4>
// CHECK-NEXT: %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr<!sycl_item_2_1_, 4>) -> memref<?x!sycl_item_2_1_, 4>
// CHECK-NEXT: sycl.constructor(%2, %5) {MangledName = @_ZN4sycl3_V12idILi2EEC1ILi2ELb1EEERNSt9enable_ifIXeqT_Li2EEKNS0_4itemILi2EXT0_EEEE4typeE, Type = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_item_2_1_, 4>) -> ()

// CHECK-LLVM: define spir_func void @cons_3([[ITEM_TYPE:%"class.sycl::_V1::item.2.true"]]* noundef byval(%"class.sycl::_V1::item.2.true") align 8 [[ARG0:%.*]]) #1
// CHECK-LLVM-DAG: [[ID:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]  
// CHECK-LLVM: [[ID_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: [[ITEM_AS:%.*]] = addrspacecast [[ITEM_TYPE]]* [[ARG0]] to [[ITEM_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ILi2ELb1EEERNSt9enable_ifIXeqT_Li2EEKNS0_4itemILi2EXT0_EEEE4typeE([[ID_TYPE]] addrspace(4)* [[ID_AS]], [[ITEM_TYPE]] addrspace(4)* [[ITEM_AS]])

extern "C" SYCL_EXTERNAL void cons_3(sycl::item<2, true> val) {
  auto id = sycl::id<2>{val};
}

// CHECK-LABEL: func.func @cons_4(%arg0: memref<?x!sycl_id_2_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_2_, llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], {{.*}}}
// CHECK-NEXT: %alloca = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %1 = llvm.addrspacecast %0 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: %3 = "polygeist.memref2pointer"(%arg0) : (memref<?x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %4 = llvm.addrspacecast %3 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: sycl.constructor(%2, %5) {MangledName = @_ZN4sycl3_V12idILi2EEC1ERKS2_, Type = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_id_2_, 4>) -> ()

// CHECK-LLVM: define spir_func void @cons_4([[ID_TYPE:%"class.sycl::_V1::id.2"]]*  noundef byval(%"class.sycl::_V1::id.2") align 8 [[ARG0:%.*]]) #1
// CHECK-LLVM-DAG: [[ID:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: [[ID2_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ARG0]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V12idILi2EEC1ERKS2_([[ID_TYPE]] addrspace(4)* [[ID1_AS]], [[ID_TYPE]] addrspace(4)* [[ID2_AS]])

extern "C" SYCL_EXTERNAL void cons_4(sycl::id<2> val) {
  auto id = sycl::id<2>{val};
}

// CHECK-LABEL: func.func @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev({{.*}})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKONCE]], {{.*}}}
// CHECK: [[I:%.*]] = "polygeist.subindex"(%arg0, %c0) : (memref<?x!sycl_accessor_1_i32_write_global_buffer, 4>, index) -> memref<?x!sycl_accessor_impl_device_1_, 4>
// CHECK: sycl.constructor([[I]], {{%.*}}, {{%.*}}, {{%.*}}) {MangledName = @_ZN4sycl3_V16detail18AccessorImplDeviceILi1EEC1ENS0_2idILi1EEENS0_5rangeILi1EEES7_, Type = @AccessorImplDevice} : (memref<?x!sycl_accessor_impl_device_1_, 4>, memref<?x!sycl_id_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>) -> ()

// CHECK-LLVM-LABEL: define spir_func void @cons_5() #1
// CHECK-LLVM: [[ACCESSOR:%.*]] = alloca %"class.sycl::_V1::accessor.1", align 8
// CHECK-LLVM: [[ACAST:%.*]] = addrspacecast %"class.sycl::_V1::accessor.1"* [[ACCESSOR]] to %"class.sycl::_V1::accessor.1" addrspace(4)*
// CHECK-LLVM: call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* [[ACAST]])

extern "C" SYCL_EXTERNAL void cons_5() {
  auto accessor = sycl::accessor<sycl::cl_int, 1, sycl::access::mode::write>{};
}

// Keep at the end.
// CHECK-LLVM: attributes #1 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="{{.*}}/Test/Verification/sycl/constructors.cpp" }
