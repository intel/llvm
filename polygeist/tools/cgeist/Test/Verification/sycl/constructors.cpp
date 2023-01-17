// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefixes=CHECK-LLVM

#include <sycl/aliases.hpp>
#include <sycl/sycl.hpp>

// CHECK-DAG: !sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
// CHECK-DAG: !sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
// CHECK-DAG: !sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
// CHECK-DAG: !sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
// CHECK-DAG: !sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
// CHECK-DAG: !sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
// CHECK-DAG: !sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
// CHECK-DAG: !sycl_accessor_1_i32_w_gb = !sycl.accessor<[1, i32, write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>
// CHECK-DAG: ![[ITEM_BASE:.*]] = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
// CHECK-DAG: ![[ITEM:.*]] = !sycl.item<[2, true], (![[ITEM_BASE]])>

// Check globals referenced in device functions are created in the GPU module
// CHECK: gpu.module @device_functions {
// CHECK-DAG: memref.global constant @__spirv_BuiltInSubgroupLocalInvocationId : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInSubgroupId : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInNumSubgroups : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInSubgroupMaxSize : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInSubgroupSize : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInLocalInvocationId : memref<vector<3xi64>, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInWorkgroupId : memref<vector<3xi64>, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInWorkgroupSize : memref<vector<3xi64>, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInNumWorkgroups : memref<vector<3xi64>, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInGlobalOffset : memref<vector<3xi64>, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInGlobalSize : memref<vector<3xi64>, 1> {alignment = 32 : i64}
// CHECK-DAG: memref.global constant @__spirv_BuiltInGlobalInvocationId : memref<vector<3xi64>, 1> {alignment = 32 : i64}

// CHECK-LLVM-DAG: @__spirv_BuiltInSubgroupLocalInvocationId = external addrspace(1) constant i32, align 4
// CHECK-LLVM-DAG: @__spirv_BuiltInSubgroupId = external addrspace(1) constant i32, align 4
// CHECK-LLVM-DAG: @__spirv_BuiltInNumSubgroups = external addrspace(1) constant i32, align 4
// CHECK-LLVM-DAG: @__spirv_BuiltInSubgroupMaxSize = external addrspace(1) constant i32, align 4
// CHECK-LLVM-DAG: @__spirv_BuiltInSubgroupSize = external addrspace(1) constant i32, align 4
// CHECK-LLVM-DAG: @__spirv_BuiltInLocalInvocationId = external addrspace(1) constant <3 x i64>, align 32
// CHECK-LLVM-DAG: @__spirv_BuiltInWorkgroupId = external addrspace(1) constant <3 x i64>, align 32
// CHECK-LLVM-DAG: @__spirv_BuiltInWorkgroupSize = external addrspace(1) constant <3 x i64>, align 32
// CHECK-LLVM-DAG: @__spirv_BuiltInNumWorkgroups = external addrspace(1) constant <3 x i64>, align 32
// CHECK-LLVM-DAG: @__spirv_BuiltInGlobalOffset = external addrspace(1) constant <3 x i64>, align 32
// CHECK-LLVM-DAG: @__spirv_BuiltInGlobalSize = external addrspace(1) constant <3 x i64>, align 32
// CHECK-LLVM-DAG: @__spirv_BuiltInGlobalInvocationId = external addrspace(1) constant <3 x i64>, align 32

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
// CHECK-LLVM-SAME:                          [[RANGE_TYPE:%"class.sycl::_V1::range.1"]]* noundef byval(%"class.sycl::_V1::range.1") align 8 [[ARG1:%.*]]) #[[FUNCATTRS:[0-9]+]]
// CHECK-LLVM-DAG: [[RANGE:%.*]] = alloca [[RANGE_TYPE]]
// CHECK-LLVM-DAG: [[ID:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call spir_func void @_ZN4sycl3_V12idILi1EEC1ERKS2_([[ID_TYPE]] addrspace(4)* [[ID1_AS]], 
// CHECK-LLVM: [[RANGE1_AS:%.*]] = addrspacecast [[RANGE_TYPE]]* [[RANGE]] to [[RANGE_TYPE]] addrspace(4)*
// CHECK-LLVM: call spir_func void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_([[RANGE_TYPE]] addrspace(4)* [[RANGE1_AS]],

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
// CHECK-NEXT: sycl.constructor(%5) {MangledFunctionName = @_ZN4sycl3_V12idILi2EEC1Ev, TypeName = @id} : (memref<?x!sycl_id_2_, 4>) -> ()

// CHECK-LLVM-LABEL: define spir_func void @cons_1()
// CHECK-LLVM-SAME:  #[[FUNCATTRS]]
// CHECK-LLVM: [[ID1:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]
// CHECK-LLVM: [[CAST1:%.*]] = bitcast [[ID_TYPE]]* %1 to i8*
// CHECK-LLVM: call void @llvm.memset.p0i8.i64(i8* %2, i8 0, i64 16, i1 false)
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID1]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call spir_func void @_ZN4sycl3_V12idILi2EEC1Ev([[ID_TYPE]] addrspace(4)* [[ID1_AS]])  

extern "C" SYCL_EXTERNAL void cons_1() {
  auto id = sycl::id<2>{};
}

// CHECK-LABEL: func.func @cons_2(%arg0: i64 {llvm.noundef}, %arg1: i64 {llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], {{.*}}}
// CHECK-NEXT: %alloca = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %1 = llvm.addrspacecast %0 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: sycl.constructor(%2, %arg0, %arg1) {MangledFunctionName = @_ZN4sycl3_V12idILi2EEC1ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm, TypeName = @id} : (memref<?x!sycl_id_2_, 4>, i64, i64) -> ()

// CHECK-LLVM-LABEL: define spir_func void @cons_2(i64 noundef %0, i64 noundef %1)
// CHECK-LLVM-SAME:  #[[FUNCATTRS]]
// CHECK-LLVM: [[ID1:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID1]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call spir_func void @_ZN4sycl3_V12idILi2EEC1ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[ID_TYPE]] addrspace(4)* [[ID1_AS]], i64 %0, i64 %1)

extern "C" SYCL_EXTERNAL void cons_2(size_t a, size_t b) {
  auto id = sycl::id<2>{a, b};
}

// CHECK-LABEL: func.func @cons_3(
// CHECK-SAME:                    %arg0: memref<?x![[ITEM]]> {llvm.align = 8 : i64, llvm.byval = ![[ITEM]], llvm.noundef})
// CHECK-SAME: attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], {{.*}}}
// CHECK-NEXT: %alloca = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT: %0 = "polygeist.memref2pointer"(%alloca) : (memref<1x!sycl_id_2_>) -> !llvm.ptr<!sycl_id_2_>
// CHECK-NEXT: %1 = llvm.addrspacecast %0 : !llvm.ptr<!sycl_id_2_> to !llvm.ptr<!sycl_id_2_, 4>
// CHECK-NEXT: %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<!sycl_id_2_, 4>) -> memref<?x!sycl_id_2_, 4>
// CHECK-NEXT: %3 = "polygeist.memref2pointer"(%arg0) : (memref<?x![[ITEM]]>) -> !llvm.ptr<![[ITEM]]>
// CHECK-NEXT: %4 = llvm.addrspacecast %3 : !llvm.ptr<![[ITEM]]> to !llvm.ptr<![[ITEM]], 4>
// CHECK-NEXT: %5 = "polygeist.pointer2memref"(%4) : (!llvm.ptr<![[ITEM]], 4>) -> memref<?x![[ITEM]], 4>
// CHECK-NEXT: sycl.constructor(%2, %5) {MangledFunctionName = @_ZN4sycl3_V12idILi2EEC1ILi2ELb1EEERNSt9enable_ifIXeqT_Li2EEKNS0_4itemILi2EXT0_EEEE4typeE, TypeName = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x![[ITEM]], 4>) -> ()

// CHECK-LLVM: define spir_func void @cons_3([[ITEM_TYPE:%"class.sycl::_V1::item.2.true"]]* noundef byval(%"class.sycl::_V1::item.2.true") align 8 [[ARG0:%.*]]) #[[FUNCATTRS]]
// CHECK-LLVM-DAG: [[ID:%.*]] = alloca [[ID_TYPE:%"class.sycl::_V1::id.2"]]  
// CHECK-LLVM: [[ID_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: [[ITEM_AS:%.*]] = addrspacecast [[ITEM_TYPE]]* [[ARG0]] to [[ITEM_TYPE]] addrspace(4)*
// CHECK-LLVM: call spir_func void @_ZN4sycl3_V12idILi2EEC1ILi2ELb1EEERNSt9enable_ifIXeqT_Li2EEKNS0_4itemILi2EXT0_EEEE4typeE([[ID_TYPE]] addrspace(4)* [[ID_AS]], [[ITEM_TYPE]] addrspace(4)* [[ITEM_AS]])

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
// CHECK-NEXT: sycl.constructor(%2, %5) {MangledFunctionName = @_ZN4sycl3_V12idILi2EEC1ERKS2_, TypeName = @id} : (memref<?x!sycl_id_2_, 4>, memref<?x!sycl_id_2_, 4>) -> ()

// CHECK-LLVM: define spir_func void @cons_4([[ID_TYPE:%"class.sycl::_V1::id.2"]]*  noundef byval(%"class.sycl::_V1::id.2") align 8 [[ARG0:%.*]]) #[[FUNCATTRS]]
// CHECK-LLVM-DAG: [[ID:%.*]] = alloca [[ID_TYPE]]
// CHECK-LLVM: [[ID1_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ID]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: [[ID2_AS:%.*]] = addrspacecast [[ID_TYPE]]* [[ARG0]] to [[ID_TYPE]] addrspace(4)*
// CHECK-LLVM: call spir_func void @_ZN4sycl3_V12idILi2EEC1ERKS2_([[ID_TYPE]] addrspace(4)* [[ID1_AS]], [[ID_TYPE]] addrspace(4)* [[ID2_AS]])

extern "C" SYCL_EXTERNAL void cons_4(sycl::id<2> val) {
  auto id = sycl::id<2>{val};
}

// CHECK-LABEL: func.func @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(
// CHECK-SAME: {{.*}}) attributes {[[SPIR_FUNCCC]], [[LINKONCE]], {{.*}}}
// CHECK: [[I:%.*]] = "polygeist.subindex"(%arg0, %c0) : (memref<?x!sycl_accessor_1_i32_w_gb, 4>, index) -> memref<?x!sycl_accessor_impl_device_1_, 4>
// CHECK: sycl.constructor([[I]], {{%.*}}, {{%.*}}, {{%.*}}) {MangledFunctionName = @_ZN4sycl3_V16detail18AccessorImplDeviceILi1EEC1ENS0_2idILi1EEENS0_5rangeILi1EEES7_, TypeName = @AccessorImplDevice} : (memref<?x!sycl_accessor_impl_device_1_, 4>, memref<?x!sycl_id_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>) -> ()

// CHECK-LLVM-LABEL: define spir_func void @cons_5()
// CHECK-LLVM-SAME:  #[[FUNCATTRS]]
// CHECK-LLVM: [[ACCESSOR:%.*]] = alloca %"class.sycl::_V1::accessor.1", align 8
// CHECK-LLVM: [[ACAST:%.*]] = addrspacecast %"class.sycl::_V1::accessor.1"* [[ACCESSOR]] to %"class.sycl::_V1::accessor.1" addrspace(4)*
// CHECK-LLVM: call spir_func void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* [[ACAST]])

extern "C" SYCL_EXTERNAL void cons_5() {
  auto accessor = sycl::accessor<sycl::cl_int, 1, sycl::access::mode::write>{};
}

// CHECK-LABEL: func.func @cons_6(
// CHECK-SAME:                    %{{.*}}: i32
// CHECK:         sycl.constructor({{.*}}, {{.*}}) {MangledFunctionName = @[[VEC_SPLAT_CTR:.*]], TypeName = @vec} : (memref<?x!sycl_vec_i32_8_, 4>, memref<?xi32, 4>) -> ()
// CHECK:         func.func @[[VEC_SPLAT_CTR]](%{{.*}}: memref<?x!sycl_vec_i32_8_, 4> {{{.*}}}, %{{.*}}: memref<?xi32, 4> {{{.*}}}) attributes {[[SPIR_FUNCCC]], [[LINKONCE]], {{.*}}}
// CHECK:         vector.splat %{{.*}} : vector<8xi32>

// CHECK-LLVM-LABEL: define spir_func void @cons_6(
// CHECK-LLVM-SAME:                                i32 noundef %{{.*}}) #[[FUNCATTRS]]
// CHECK-LLVM:         call spir_func void @[[VEC_SPLAT_CTR:.*]](%"class.sycl::_V1::vec" addrspace(4)* %{{.*}}, i32 addrspace(4)* %{{.*}})
// CHECK-LLVM:       define linkonce_odr spir_func void @[[VEC_SPLAT_CTR]](%"class.sycl::_V1::vec" addrspace(4)* noundef align 32 %{{.*}}, i32 addrspace(4)* noundef align 4 %{{.*}}) #[[FUNCATTRS]] {
// CHECK-LLVM:         %[[VECINIT:.*]] = insertelement <8 x i32> undef, i32 %{{.*}}, i32 0
// CHECK-LLVM:         %{{.*}} = shufflevector <8 x i32> %[[VECINIT]], <8 x i32> undef, <8 x i32> zeroinitializer

extern "C" SYCL_EXTERNAL void cons_6(int Arg) {
  auto vec = sycl::vec<sycl::cl_int, 8>{Arg};
}

// CHECK-LABEL: func.func @cons_7(
// CHECK-SAME:                    %[[ARG0:.*]]: f32 {{{.*}}}, %[[ARG1:.*]]: f32 {{{.*}}}, %[[ARG2:.*]]: f32 {{{.*}}}, %[[ARG3:.*]]: f32 {{{.*}}})
// CHECK:         sycl.constructor(%{{.*}}, %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) {MangledFunctionName = @[[VEC_INITLIST_CTR:.*]], TypeName = @vec} : (memref<?x!sycl_vec_f32_4_, 4>, f32, f32, f32, f32) -> ()
// CHECK:       func.func @[[VEC_INITLIST_CTR]](%{{.*}}: memref<?x!sycl_vec_f32_4_, 4> {{{.*}}}, %{{.*}}: f32 {{{.*}}}, %{{.*}}: f32 {{{.*}}}, %{{.*}}: f32 {{{.*}}}, %{{.*}}: f32 {{{.*}}}) attributes {[[SPIR_FUNCCC]], [[LINKONCE]], {{.*}}}

// CHECK-LLVM-LABEL: define spir_func void @cons_7(
// CHECK-LLVM-SAME:                                float noundef %[[ARG0:.*]], float noundef %[[ARG1:.*]], float noundef %[[ARG2:.*]], float noundef %[[ARG3:.*]]) #[[FUNCATTRS]]
// CHECK-LLVM:         call spir_func void @[[VEC_INITLIST_CTR:.*]](%"class.sycl::_V1::vec.1" addrspace(4)* %{{.*}}, float %[[ARG0]], float %[[ARG1]], float %[[ARG2]], float %[[ARG3]])
// CHECK-LLVM:       define linkonce_odr spir_func void @[[VEC_INITLIST_CTR]](%"class.sycl::_V1::vec.1" addrspace(4)* noundef align 16 {{.*}}, float noundef {{.*}}, float noundef {{.*}}, float noundef {{.*}}, float noundef {{.*}}) #[[FUNCATTRS]]
extern "C" SYCL_EXTERNAL void cons_7(float A, float B, float C, float D) {
  auto vec = sycl::vec<sycl::cl_float, 4>{A, B, C, D};
}

// CHECK-LABEL: func.func @cons_8(
// CHECK-SAME:                    %[[ARG0:.*]]: memref<?x!sycl_vec_f64_16_, 4> {{{.*}}})
// CHECK:         sycl.constructor(%{{.*}}, %[[ARG0]]) {MangledFunctionName = @[[VEC_COPY_CTR:.*]], TypeName = @vec} : (memref<?x!sycl_vec_f64_16_, 4>, memref<?x!sycl_vec_f64_16_, 4>) -> ()
// CHECK:       func.func @[[VEC_COPY_CTR]](%{{.*}}: memref<?x!sycl_vec_f64_16_, 4> {{{.*}}}, %{{.*}}: memref<?x!sycl_vec_f64_16_, 4> {{{.*}}}) attributes {[[SPIR_FUNCCC]], [[LINKONCE]], {{.*}}}

// CHECK-LLVM-LABEL:  define spir_func void @cons_8(
// CHECK-LLVM-SAME:                                 %"class.sycl::_V1::vec.2" addrspace(4)* noundef align 128 %[[ARG0:.*]]) #[[FUNCATTRS]] {
// CHECK-LLVM:          call spir_func void @_ZN4sycl3_V13vecIdLi16EEC1ERKS2_(%"class.sycl::_V1::vec.2" addrspace(4)* %{{.*}}, %"class.sycl::_V1::vec.2" addrspace(4)* %[[ARG0]])
// CHECK-LLVM:        define linkonce_odr spir_func void @_ZN4sycl3_V13vecIdLi16EEC1ERKS2_(%"class.sycl::_V1::vec.2" addrspace(4)* noundef align 128 %{{.*}}, %{{.*}}class.sycl::_V1::vec.2" addrspace(4)* noundef align 128 %{{.*}}) #[[FUNCATTRS]] {
extern "C" SYCL_EXTERNAL void cons_8(const sycl::vec<sycl::cl_double, 16> &Other) {
  auto vec = sycl::vec<sycl::cl_double, 16>{Other};
}

// CHECK-LABEL: func.func @cons_9(
// CHECK-SAME:                    %[[ARG0:.*]]: vector<3xi8>
// CHECK:         sycl.constructor(%{{.*}}, %[[ARG0]]) {MangledFunctionName = @[[VEC_NATIVE_CTR:.*]], TypeName = @vec} : (memref<?x!sycl_vec_i8_3_, 4>, vector<3xi8>) -> ()
// CHECK:       func.func @[[VEC_NATIVE_CTR]](%{{.*}}: memref<?x!sycl_vec_i8_3_, 4> {{{.*}}}, %{{.*}}: vector<3xi8> {{{.*}}}) attributes {[[SPIR_FUNCCC]], [[LINKONCE]], {{.*}}}

// CHECK-LLVM-LABEL:  define spir_func void @cons_9(
// CHECK-LLVM-SAME:                                 <3 x i8> noundef %[[ARG0:.*]]) #[[FUNCATTRS]] {
// CHECK-LLVM:          call spir_func void @[[VEC_NATIVE_CTR:.*]](%"class.sycl::_V1::vec.3" addrspace(4)* %{{.*}}, <3 x i8> %[[ARG0]])
// CHECK-LLVM:        define linkonce_odr spir_func void @[[VEC_NATIVE_CTR]](%"class.sycl::_V1::vec.3" addrspace(4)* noundef align 4 %{{.*}}, <3 x i8> noundef %{{.*}}) #[[FUNCATTRS]] {
extern "C" SYCL_EXTERNAL void cons_9(const sycl::vec<sycl::cl_char, 3>::vector_t Native) {
  auto vec = sycl::vec<sycl::cl_char, 3>{Native};
}

// CHECK-LABEL: func.func @cons_10(
// CHECK-SAME:                     %[[ARG0:.*]]: memref<?x!sycl_vec_i64_8_, 4> {{{.*}}}, %[[ARG1:.*]]: memref<?x!sycl_vec_i64_4_, 4> {{{.*}}}, %[[ARG2:.*]]: memref<?x!sycl_vec_i64_2_, 4> {{{.*}}}, %{{.*}}: i64 {{{.*}}}, %{{.*}}: i64 {{{.*}}}) attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], {{.*}}}
// CHECK:         sycl.constructor(%3, %[[ARG0]], %[[ARG1]], %[[ARG2]], %6, %9) {MangledFunctionName = @[[VEC_INITLIST_VEC_CTR:.*]], TypeName = @vec} : (memref<?x!sycl_vec_i64_16_, 4>, memref<?x!sycl_vec_i64_8_, 4>, memref<?x!sycl_vec_i64_4_, 4>, memref<?x!sycl_vec_i64_2_, 4>, memref<?xi64, 4>, memref<?xi64, 4>) -> ()
// CHECK:       func.func @[[VEC_INITLIST_VEC_CTR]](%{{.*}}: memref<?x!sycl_vec_i64_16_, 4> {{{.*}}}, %{{.*}}: memref<?x!sycl_vec_i64_8_, 4> {{{.*}}}, %{{.*}}: memref<?x!sycl_vec_i64_4_, 4> {{{.*}}}, %{{.*}}: memref<?x!sycl_vec_i64_2_, 4> {{{.*}}}, %{{.*}}: memref<?xi64, 4> {{{.*}}}, %{{.*}}: memref<?xi64, 4> {{{.*}}}) attributes {[[SPIR_FUNCCC]], [[LINKONCE]], {{.*}}}

// CHECK-LLVM-LABEL: define spir_func void @cons_10(
// CHECK-LLVM-SAME:                                 %"class.sycl::_V1::vec.5" addrspace(4)* noundef align 64 %[[ARG0:.*]], %"class.sycl::_V1::vec.6" addrspace(4)* noundef align 32 %[[ARG1:.*]], %"class.sycl::_V1::vec.7" addrspace(4)* noundef align 16 %[[ARG2:.*]], i64 noundef %{{.*}}, i64 noundef %{{.*}}) #[[FUNCATTRS]] {
// CHECK-LLVM:         call spir_func void @[[VEC_INITLIST_VEC_CTR:.*]](%"class.sycl::_V1::vec.4" addrspace(4)* %{{.*}}, %"class.sycl::_V1::vec.5" addrspace(4)* %[[ARG0]], %"class.sycl::_V1::vec.6" addrspace(4)* %[[ARG1]], %"class.sycl::_V1::vec.7" addrspace(4)* %[[ARG2]], i64 addrspace(4)* %{{.*}}, i64 addrspace(4)* %{{.*}})
// CHECK-LLVM:       define linkonce_odr spir_func void @[[VEC_INITLIST_VEC_CTR]](%"class.sycl::_V1::vec.4" addrspace(4)* noundef align 128 %{{.*}}, %"class.sycl::_V1::vec.5" addrspace(4)* noundef align 64 %{{.*}}, %"class.sycl::_V1::vec.6" addrspace(4)* noundef align 32 %{{.*}}, %"class.sycl::_V1::vec.7" addrspace(4)* noundef align 16 %{{.*}}, i64 addrspace(4)* noundef align 8 %{{.*}}, i64 addrspace(4)* noundef align 8 %{{.*}}) #[[FUNCATTRS]] {

extern "C" SYCL_EXTERNAL void cons_10(const sycl::long8 &A,
				      const sycl::long4 &B,
				      const sycl::long2 &C,
				      sycl::cl_long D,
				      sycl::cl_long E) {
  auto vec = sycl::long16{A, B, C, D, E};
}

// CHECK-LABEL: func.func @cons_11()
// CHECK-SAME:                       attributes {[[SPIR_FUNCCC]], [[LINKEXTERNAL]], {{.*}}}
// CHECK-DAG:     %[[FALSE:.*]] = arith.constant false
// CHECK-DAG:     %[[C0_I8:.*]] = arith.constant 0 : i8
// CHECK-NEXT:    %[[ALLOCA:.*]] = memref.alloca() : memref<1x!sycl_vec_i32_4_>
// CHECK-NEXT:    %[[VAL0:.*]] = "polygeist.memref2pointer"(%[[ALLOCA]]) : (memref<1x!sycl_vec_i32_4_>) -> !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL1:.*]] = "polygeist.typeSize"() {source = !sycl_vec_i32_4_} : () -> index
// CHECK-NEXT:    %[[VAL2:.*]] = arith.index_cast %[[VAL1]] : index to i64
// CHECK-NEXT:    "llvm.intr.memset"(%[[VAL0]], %[[C0_I8]], %[[VAL2]], %[[FALSE]]) : (!llvm.ptr<i8>, i8, i64, i1) -> ()

// CHECK-LLVM-LABEL:  define spir_func void @cons_11()
// CHECK-LLVM-SAME:                                    #[[FUNCATTRS]] {
// CHECK-LLVM:          %[[VAL1:.*]] = alloca %"class.sycl::_V1::vec.8", align 16
// CHECK-LLVM:          %[[VAL2:.*]] = bitcast %"class.sycl::_V1::vec.8"* %[[VAL1]] to i8*
// CHECK-LLVM:          call void @llvm.memset.p0i8.i64(i8* %[[VAL2]], i8 0, i64 16, i1 false)

extern "C" SYCL_EXTERNAL void cons_11() {
  auto vec = sycl::vec<sycl::cl_int, 4>{};
}

// Keep at the end.
// CHECK-LLVM: attributes #[[FUNCATTRS]] = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="{{.*}}/Test/Verification/sycl/constructors.cpp" }
