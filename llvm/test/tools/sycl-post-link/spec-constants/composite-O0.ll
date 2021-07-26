; RUN: sycl-post-link -spec-const=rt --ir-output-only %s -S -o - \
; RUN: | FileCheck %s --implicit-check-not "call {{.*}} __sycl_getCompositeSpecConstantValue"
;
; This test is intended to check that sycl-post-link tool is capable of handling
; composite specialization constants by lowering them into a set of SPIR-V
; friendly IR operations representing those constants.
; This particular LLVM IR is generated from the same source as for
; composite-spec-constant.ll test, but -O0 optimization level was used to check
; that sycl-post-link is capable to handle this form of LLVM IR as well.
;
; CHECK: %[[#NS0:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID:]], i32
; CHECK: %[[#NS1:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#ID + 1]], float
; CHECK: %[[#NA0:]] = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %[[#NS0]], float %[[#NS1]])
;
; CHECK: %[[#NS2:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID + 2]], i32
; CHECK: %[[#NS3:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#ID + 3]], float
; CHECK: %[[#NA1:]] = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %[[#NS2]], float %[[#NS3]])
;
; CHECK: %[[#NA:]] = call [2 x %struct._ZTS1A.A] @_Z29__spirv_SpecConstantCompositestruct._ZTS1A.Astruct._ZTS1A.A(%struct._ZTS1A.A %[[#NA0]], %struct._ZTS1A.A %[[#NA1]])
;
; CHECK: %[[#B:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID + 4]], i32{{.*}})
;
; CHECK: %[[#POD:]] = call %struct._ZTS3POD.POD @_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Ai([2 x %struct._ZTS1A.A] %[[#NA]], i32 %[[#B]])
; CHECK: store %struct._ZTS3POD.POD %[[#POD]]
;
; CHECK: !sycl.specialization-constants = !{![[#MD:]]}
; CHECK: ![[#MD]] = !{!"_ZTS3POD", i32 [[#ID]], i32 0, i32 4,
; CHECK-SAME: i32 [[#ID + 1]], i32 4, i32 4,
; CHECK-SAME: i32 [[#ID + 2]], i32 8, i32 4,
; CHECK-SAME: i32 [[#ID + 3]], i32 12, i32 4,
; CHECK-SAME: i32 [[#ID + 4]], i32 16, i32 4}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%struct._ZTS3POD.POD = type { [2 x %struct._ZTS1A.A], i32 }
%struct._ZTS1A.A = type { i32, float }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" = type <{ %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant", [7 x i8] }>
%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" = type { %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %union._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEUt_E.anon }
%"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" = type { %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" }
%union._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEUt_E.anon = type { %struct._ZTS3POD.POD addrspace(1)* }
%"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" = type { i8 }
%"class._ZTSN2cl4sycl6detail15accessor_commonI3PODLi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::detail::accessor_common" = type { i8 }

$_ZTS4Test = comdat any

$_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev = comdat any

$_ZN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EC2Ev = comdat any

$_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1S2_NS0_5rangeILi1EEESE_NS0_2idILi1EEE = comdat any

$_ZN2cl4sycl2idILi1EEC2Ev = comdat any

$_ZN2cl4sycl6detail14InitializedValILi1ENS0_5rangeEE3getILi0EEENS3_ILi1EEEv = comdat any

$_ZN2cl4sycl6detail18AccessorImplDeviceILi1EEC2ENS0_2idILi1EEENS0_5rangeILi1EEES7_ = comdat any

$_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE = comdat any

$_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE = comdat any

$_ZN2cl4sycl6detail5arrayILi1EEixEi = comdat any

$_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE9getOffsetEv = comdat any

$_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getAccessRangeEv = comdat any

$_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getMemoryRangeEv = comdat any

$_ZNK2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_E3getIS4_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podIS8_EE5valueES8_E4typeEv = comdat any

$_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERS2_NS0_2idILi1EEE = comdat any

$_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE = comdat any

$_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEE = comdat any

$_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE15getQualifiedPtrEv = comdat any

@__builtin_unique_stable_name._ZNK2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_E3getIS4_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podIS8_EE5valueES8_E4typeEv = private unnamed_addr addrspace(1) constant [9 x i8] c"_ZTS3POD\00", align 1

; Function Attrs: convergent noinline norecurse optnone mustprogress
define weak_odr dso_local spir_kernel void @_ZTS4Test(%struct._ZTS3POD.POD addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %_arg_.addr = alloca %struct._ZTS3POD.POD addrspace(1)*, align 8
  %0 = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", align 8
  %agg.tmp = alloca %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", align 8
  %agg.tmp4 = alloca %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", align 8
  %agg.tmp5 = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", align 8
  store %struct._ZTS3POD.POD addrspace(1)* %_arg_, %struct._ZTS3POD.POD addrspace(1)** %_arg_.addr, align 8
  %1 = getelementptr inbounds %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0, i32 0, i32 0
  %2 = addrspacecast %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor"* %1 to %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*
  call spir_func void @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %2) #8
  %3 = getelementptr inbounds %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0, i32 0, i32 1
  %4 = addrspacecast %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant"* %3 to %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)*
  call spir_func void @_ZN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EC2Ev(%"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)* %4) #8
  %5 = getelementptr inbounds %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0, i32 0, i32 0
  %6 = load %struct._ZTS3POD.POD addrspace(1)*, %struct._ZTS3POD.POD addrspace(1)** %_arg_.addr, align 8
  %7 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %agg.tmp to i8*
  %8 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %7, i8* align 8 %8, i64 8, i1 false)
  %9 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %agg.tmp4 to i8*
  %10 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %9, i8* align 8 %10, i64 8, i1 false)
  %11 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %agg.tmp5 to i8*
  %12 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %11, i8* align 8 %12, i64 8, i1 false)
  %13 = addrspacecast %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor"* %5 to %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*
  call spir_func void @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1S2_NS0_5rangeILi1EEESE_NS0_2idILi1EEE(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %13, %struct._ZTS3POD.POD addrspace(1)* %6, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %agg.tmp, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %agg.tmp4, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %agg.tmp5) #8
  %14 = addrspacecast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*
  call spir_func void @"_ZZZ4mainENK3$_1clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %14) #8
  ret void
}

; Function Attrs: convergent noinline norecurse optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %agg.tmp = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", align 8
  %agg.tmp2 = alloca %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", align 8
  %agg.tmp3 = alloca %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", align 8
  store %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail15accessor_commonI3PODLi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::detail::accessor_common" addrspace(4)*
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %1 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %agg.tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  %2 = addrspacecast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %agg.tmp to %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)*
  call spir_func void @_ZN2cl4sycl2idILi1EEC2Ev(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %2) #8
  %3 = addrspacecast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %agg.tmp2 to %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)*
  call spir_func void @_ZN2cl4sycl6detail14InitializedValILi1ENS0_5rangeEE3getILi0EEENS3_ILi1EEEv(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %3) #8
  %4 = addrspacecast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %agg.tmp3 to %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)*
  call spir_func void @_ZN2cl4sycl6detail14InitializedValILi1ENS0_5rangeEE3getILi0EEENS3_ILi1EEEv(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %4) #8
  call spir_func void @_ZN2cl4sycl6detail18AccessorImplDeviceILi1EEC2ENS0_2idILi1EEENS0_5rangeILi1EEES7_(%"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %agg.tmp, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %agg.tmp2, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %agg.tmp3) #8
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EC2Ev(%"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)* %this) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)* %this, %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)*, %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)** %this.addr, align 8
  ret void
}

; Function Attrs: convergent noinline norecurse optnone mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1S2_NS0_5rangeILi1EEESE_NS0_2idILi1EEE(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %struct._ZTS3POD.POD addrspace(1)* %Ptr, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %AccessRange, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %MemRange, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %Offset) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %Ptr.addr = alloca %struct._ZTS3POD.POD addrspace(1)*, align 8
  %I = alloca i32, align 4
  store %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  store %struct._ZTS3POD.POD addrspace(1)* %Ptr, %struct._ZTS3POD.POD addrspace(1)** %Ptr.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %0 = load %struct._ZTS3POD.POD addrspace(1)*, %struct._ZTS3POD.POD addrspace(1)** %Ptr.addr, align 8
  %1 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 1
  %MData = bitcast %union._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEUt_E.anon addrspace(4)* %1 to %struct._ZTS3POD.POD addrspace(1)* addrspace(4)*
  store %struct._ZTS3POD.POD addrspace(1)* %0, %struct._ZTS3POD.POD addrspace(1)* addrspace(4)* %MData, align 8
  store i32 0, i32* %I, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %I, align 4
  %cmp = icmp slt i32 %2, 1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %Offset to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"*
  %4 = load i32, i32* %I, align 4
  %5 = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"* %3 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %call = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %5, i32 %4) #8
  %6 = load i64, i64 addrspace(4)* %call, align 8
  %call2 = call spir_func align 8 dereferenceable(8) %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1) #8
  %7 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %call2 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %8 = load i32, i32* %I, align 4
  %call3 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %7, i32 %8) #8
  store i64 %6, i64 addrspace(4)* %call3, align 8
  %9 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %AccessRange to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"*
  %10 = load i32, i32* %I, align 4
  %11 = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"* %9 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %call4 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %11, i32 %10) #8
  %12 = load i64, i64 addrspace(4)* %call4, align 8
  %call5 = call spir_func align 8 dereferenceable(8) %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getAccessRangeEv(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1) #8
  %13 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %call5 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %14 = load i32, i32* %I, align 4
  %call6 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %13, i32 %14) #8
  store i64 %12, i64 addrspace(4)* %call6, align 8
  %15 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %MemRange to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"*
  %16 = load i32, i32* %I, align 4
  %17 = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"* %15 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %call7 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %17, i32 %16) #8
  %18 = load i64, i64 addrspace(4)* %call7, align 8
  %call8 = call spir_func align 8 dereferenceable(8) %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1) #8
  %19 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %call8 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %20 = load i32, i32* %I, align 4
  %call9 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %19, i32 %20) #8
  store i64 %18, i64 addrspace(4)* %call9, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %21 = load i32, i32* %I, align 4
  %inc = add nsw i32 %21, 1
  store i32 %inc, i32* %I, align 4
  br label %for.cond, !llvm.loop !5

for.end:                                          ; preds = %for.cond
  %22 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %Offset to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"*
  %23 = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"* %22 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %call10 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %23, i32 0) #8
  %24 = load i64, i64 addrspace(4)* %call10, align 8
  %25 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 1
  %MData11 = bitcast %union._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEUt_E.anon addrspace(4)* %25 to %struct._ZTS3POD.POD addrspace(1)* addrspace(4)*
  %26 = load %struct._ZTS3POD.POD addrspace(1)*, %struct._ZTS3POD.POD addrspace(1)* addrspace(4)* %MData11, align 8
  %add.ptr = getelementptr inbounds %struct._ZTS3POD.POD, %struct._ZTS3POD.POD addrspace(1)* %26, i64 %24
  store %struct._ZTS3POD.POD addrspace(1)* %add.ptr, %struct._ZTS3POD.POD addrspace(1)* addrspace(4)* %MData11, align 8
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #4

; Function Attrs: convergent noinline norecurse optnone mustprogress
define internal spir_func void @"_ZZZ4mainENK3$_1clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this) #3 align 2 {
entry:
  %this.addr = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, align 8
  %ref.tmp = alloca %struct._ZTS3POD.POD, align 4
  %agg.tmp = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", align 8
  store %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, align 8
  %0 = getelementptr inbounds %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this1, i32 0, i32 1
  %1 = addrspacecast %struct._ZTS3POD.POD* %ref.tmp to %struct._ZTS3POD.POD addrspace(4)*
  call spir_func void @_ZNK2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_E3getIS4_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podIS8_EE5valueES8_E4typeEv(%struct._ZTS3POD.POD addrspace(4)* sret(%struct._ZTS3POD.POD) align 4 %1, %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)* %0) #8
  %2 = getelementptr inbounds %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this1, i32 0, i32 0
  %3 = addrspacecast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %agg.tmp to %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)*
  call spir_func void @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %3, i64 0) #8
  %call = call spir_func align 4 dereferenceable(20) %struct._ZTS3POD.POD addrspace(4)* @_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERS2_NS0_2idILi1EEE(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %agg.tmp) #8
  %4 = bitcast %struct._ZTS3POD.POD addrspace(4)* %call to i8 addrspace(4)*
  %5 = bitcast %struct._ZTS3POD.POD* %ref.tmp to i8*
  call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 4 %4, i8* align 4 %5, i64 20, i1 false)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #5

; Function Attrs: convergent noinline norecurse optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl2idILi1EEC2Ev(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %this, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)** %this.addr, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  call spir_func void @_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %0, i64 0) #8
  ret void
}

; Function Attrs: convergent noinline norecurse optnone mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail14InitializedValILi1ENS0_5rangeEE3getILi0EEENS3_ILi1EEEv(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %agg.result) #3 comdat align 2 {
entry:
  call spir_func void @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %agg.result, i64 0) #8
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail18AccessorImplDeviceILi1EEC2ENS0_2idILi1EEENS0_5rangeILi1EEES7_(%"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %Offset, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %AccessRange, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %MemoryRange) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this, %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)*, %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)** %this.addr, align 8
  %Offset2 = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 0
  %0 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %Offset2 to i8 addrspace(4)*
  %1 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %Offset to i8*
  call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 8 %0, i8* align 8 %1, i64 8, i1 false)
  %AccessRange3 = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 1
  %2 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %AccessRange3 to i8 addrspace(4)*
  %3 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %AccessRange to i8*
  call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 8 %2, i8* align 8 %3, i64 8, i1 false)
  %MemRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 2
  %4 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %MemRange to i8 addrspace(4)*
  %5 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %MemoryRange to i8*
  call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 8 %4, i8* align 8 %5, i64 8, i1 false)
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %this, i64 %dim0) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %dim0.addr = alloca i64, align 8
  store %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %this, %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)** %this.addr, align 8
  store i64 %dim0, i64* %dim0.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)** %this.addr, align 8
  %common_array = getelementptr inbounds %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array", %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %this1, i32 0, i32 0
  %arrayinit.begin = getelementptr inbounds [1 x i64], [1 x i64] addrspace(4)* %common_array, i64 0, i64 0
  %0 = load i64, i64* %dim0.addr, align 8
  store i64 %0, i64 addrspace(4)* %arrayinit.begin, align 8
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %this, i64 %dim0) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)*, align 8
  %dim0.addr = alloca i64, align 8
  store %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %this, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)** %this.addr, align 8
  store i64 %dim0, i64* %dim0.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)** %this.addr, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %1 = load i64, i64* %dim0.addr, align 8
  call spir_func void @_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %0, i64 %1) #8
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #4

; Function Attrs: convergent noinline norecurse optnone mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %this, i32 %dimension) #3 comdat align 2 {
entry:
  %this.addr.i = alloca %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %dimension.addr.i = alloca i32, align 4
  %this.addr = alloca %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %dimension.addr = alloca i32, align 4
  store %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %this, %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)** %this.addr, align 8
  store i32 %dimension, i32* %dimension.addr, align 4
  %this1 = load %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)** %this.addr, align 8
  %0 = load i32, i32* %dimension.addr, align 4
  store %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %this1, %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)** %this.addr.i, align 8
  store i32 %0, i32* %dimension.addr.i, align 4
  %this1.i = load %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)** %this.addr.i, align 8
  %common_array = getelementptr inbounds %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array", %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %this1, i32 0, i32 0
  %1 = load i32, i32* %dimension.addr, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [1 x i64], [1 x i64] addrspace(4)* %common_array, i64 0, i64 %idxprom
  ret i64 addrspace(4)* %arrayidx
}

; Function Attrs: convergent noinline norecurse nounwind optnone mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this) #6 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %Offset = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 0
  ret %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %Offset
}

; Function Attrs: convergent noinline norecurse nounwind optnone mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getAccessRangeEv(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this) #6 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %AccessRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 1
  ret %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %AccessRange
}

; Function Attrs: convergent noinline norecurse nounwind optnone mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this) #6 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %MemRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 2
  ret %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" addrspace(4)* %MemRange
}

; Function Attrs: convergent noinline norecurse optnone mustprogress
define linkonce_odr dso_local spir_func void @_ZNK2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_E3getIS4_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podIS8_EE5valueES8_E4typeEv(%struct._ZTS3POD.POD addrspace(4)* noalias sret(%struct._ZTS3POD.POD) align 4 %agg.result, %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)* %this) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)*, align 8
  %TName = alloca i8 addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)* %this, %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)*, %"class._ZTSN2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_EE.cl::sycl::ext::oneapi::experimental::spec_constant" addrspace(4)** %this.addr, align 8
  store i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([9 x i8], [9 x i8] addrspace(1)* @__builtin_unique_stable_name._ZNK2cl4sycl3ext6oneapi12experimental13spec_constantI3PODS4_E3getIS4_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podIS8_EE5valueES8_E4typeEv, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)** %TName, align 8
  %0 = load i8 addrspace(4)*, i8 addrspace(4)** %TName, align 8
  call spir_func void @_Z36__sycl_getCompositeSpecConstantValueI3PODET_PKc(%struct._ZTS3POD.POD addrspace(4)* sret(%struct._ZTS3POD.POD) align 4 %agg.result, i8 addrspace(4)* %0) #8
  ret void
}

; Function Attrs: convergent noinline norecurse optnone mustprogress
define linkonce_odr dso_local spir_func align 4 dereferenceable(20) %struct._ZTS3POD.POD addrspace(4)* @_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERS2_NS0_2idILi1EEE(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %Index) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %LinearIndex = alloca i64, align 8
  %agg.tmp = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", align 8
  store %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %agg.tmp to i8*
  %1 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %Index to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 8, i1 false)
  %call = call spir_func i64 @_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %agg.tmp) #8
  store i64 %call, i64* %LinearIndex, align 8
  %call2 = call spir_func %struct._ZTS3POD.POD addrspace(1)* @_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE15getQualifiedPtrEv(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1) #8
  %2 = load i64, i64* %LinearIndex, align 8
  %ptridx = getelementptr inbounds %struct._ZTS3POD.POD, %struct._ZTS3POD.POD addrspace(1)* %call2, i64 %2
  %ptridx.ascast = addrspacecast %struct._ZTS3POD.POD addrspace(1)* %ptridx to %struct._ZTS3POD.POD addrspace(4)*
  ret %struct._ZTS3POD.POD addrspace(4)* %ptridx.ascast
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %this, i64 %dim0) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)*, align 8
  %dim0.addr = alloca i64, align 8
  store %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %this, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)** %this.addr, align 8
  store i64 %dim0, i64* %dim0.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)** %this.addr, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %1 = load i64, i64* %dim0.addr, align 8
  call spir_func void @_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %0, i64 %1) #8
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z36__sycl_getCompositeSpecConstantValueI3PODET_PKc(%struct._ZTS3POD.POD addrspace(4)* sret(%struct._ZTS3POD.POD) align 4, i8 addrspace(4)*) #7

; Function Attrs: convergent noinline norecurse optnone mustprogress
define linkonce_odr dso_local spir_func i64 @_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %Id) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %Result = alloca i64, align 8
  store %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %Id to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"*
  %1 = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array"* %0 to %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)*
  %call = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" addrspace(4)* %1, i32 0) #8
  %2 = load i64, i64 addrspace(4)* %call, align 8
  ret i64 %2
}

; Function Attrs: convergent noinline norecurse nounwind optnone mustprogress
define linkonce_odr dso_local spir_func %struct._ZTS3POD.POD addrspace(1)* @_ZNK2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE15getQualifiedPtrEv(%"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this) #6 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %this1 = load %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr, align 8
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 1
  %MData = bitcast %union._ZTSN2cl4sycl8accessorI3PODLi1ELNS0_6access4modeE1025ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEUt_E.anon addrspace(4)* %0 to %struct._ZTS3POD.POD addrspace(1)* addrspace(4)*
  %1 = load %struct._ZTS3POD.POD addrspace(1)*, %struct._ZTS3POD.POD addrspace(1)* addrspace(4)* %MData, align 8
  ret %struct._ZTS3POD.POD addrspace(1)* %1
}

attributes #0 = { convergent noinline norecurse optnone mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="./composite.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline norecurse optnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent noinline norecurse nounwind optnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent noinline norecurse optnone mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nofree nosync nounwind willreturn }
attributes #5 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #6 = { convergent noinline norecurse nounwind optnone mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { convergent }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0 "}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!5 = distinct !{!5, !6, !7}
!6 = !{!"llvm.loop.mustprogress"}
!7 = !{!"llvm.loop.unroll.enable"}
