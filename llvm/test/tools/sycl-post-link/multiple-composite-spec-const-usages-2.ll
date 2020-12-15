; RUN: sycl-post-link -spec-const=rt --ir-output-only %s -S -o - \
; RUN: | FileCheck %s --implicit-check-not __sycl_getCompositeSpecConstantValue
;
; This test is intended to check that sycl-post-link tool is capable of handling
; situations when the same composite specialization constants is used more than
; once. Unlike multiple-composite-spec-const-usages.ll test, this is a real life
; LLVM IR example
;
; CHECK-LABEL: @_ZTSN4test8kernel_tIfEE
; CHECK: %[[#X1:]] = call float @_Z20__spirv_SpecConstantif(i32 0, float 0
; CHECK: %[[#Y1:]] = call float @_Z20__spirv_SpecConstantif(i32 1, float 0
; CHECK: call {{.*}} @_Z29__spirv_SpecConstantCompositeff(float %[[#X1]], float %[[#Y1]]), !SYCL_SPEC_CONST_SYM_ID ![[#ID:]]
; CHECK-LABEL: @_ZNK4test8kernel_tIiEclEN2cl4sycl2idILi1EEE
; CHECK: %[[#X2:]] = call float @_Z20__spirv_SpecConstantif(i32 0, float 0
; CHECK: %[[#Y2:]] = call float @_Z20__spirv_SpecConstantif(i32 1, float 0
; CHECK: call {{.*}} @_Z29__spirv_SpecConstantCompositeff(float %[[#X2]], float %[[#Y2]]), !SYCL_SPEC_CONST_SYM_ID ![[#ID]]
; CHECK: ![[#ID]] = !{!"_ZTS11sc_kernel_t", i32 0, i32 1}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" = type { [16 x i8], %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", i32, i32, i64, i32, i32, i32, i32 }
%"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" = type { %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %union._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon }
%"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice" = type { %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%union._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon = type { i32 addrspace(1)* }
%"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" = type { %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE.cl::sycl::detail::AccessorImplDevice", %union._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon }
%union._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon = type { i8 addrspace(1)* }
%"class._ZTSN4test8kernel_tIfEE.test::kernel_t" = type { %"class._ZTSN2cl4sycl6ONEAPI12experimental13spec_constantIN4test5pod_tE11sc_kernel_tEE.cl::sycl::ONEAPI::experimental::spec_constant", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" }
%"class._ZTSN2cl4sycl6ONEAPI12experimental13spec_constantIN4test5pod_tE11sc_kernel_tEE.cl::sycl::ONEAPI::experimental::spec_constant" = type { [8 x i8] }
%"struct._ZTSN4test5pod_tE.test::pod_t" = type { float, float }

$_ZTSN4test8kernel_tIfEE = comdat any

$_ZNK4test8kernel_tIfEclEN2cl4sycl2idILi1EEE = comdat any

$_ZN2cl4sycl6stream10__finalizeEv = comdat any

$_ZN2cl4sycllsERKNS0_6streamERKf = comdat any

$_ZN2cl4sycl6detail14checkForInfNanIfEENSt9enable_ifIXoosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valueEjE4typeEPcS4_ = comdat any

$_ZN2cl4sycl6detail21floatingPointToDecStrIfEENSt9enable_ifIXoooosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valuesr3std7is_sameIS4_NS1_9half_impl4halfEEE5valueEjE4typeES4_Pcib = comdat any

$_ZTSN4test8kernel_tIiEE = comdat any

$_ZNK4test8kernel_tIiEclEN2cl4sycl2idILi1EEE = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@.str = private unnamed_addr addrspace(1) constant [11 x i8] c"--------> \00", align 1
@__builtin_unique_stable_name._ZNK2cl4sycl6ONEAPI12experimental13spec_constantIN4test5pod_tE11sc_kernel_tE3getIS5_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podISA_EE5valueESA_E4typeEv = private unnamed_addr addrspace(1) constant [18 x i8] c"_ZTS11sc_kernel_t\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [2 x i8] c"\0A\00", align 1
@.str.2 = private unnamed_addr addrspace(1) constant [4 x i8] c"nan\00", align 1
@.str.3 = private unnamed_addr addrspace(1) constant [5 x i8] c"-inf\00", align 1
@.str.4 = private unnamed_addr addrspace(1) constant [4 x i8] c"inf\00", align 1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSN4test8kernel_tIfEE(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* byval(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream") align 8 %_arg_strm_, i8 addrspace(1)* %_arg_GlobalBuf, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalBuf1, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalBuf2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalBuf3, i32 addrspace(1)* %_arg_GlobalOffset, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalOffset4, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalOffset5, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalOffset6, i8 addrspace(1)* %_arg_GlobalFlushBuf, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalFlushBuf7, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalFlushBuf8, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalFlushBuf9) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %kernel_t = alloca %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", align 8
  %agg.tmp22 = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", align 8
  %0 = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 0, i32 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 152, i8* nonnull %0) #8
  %strm_ = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 1
  %1 = getelementptr %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* %strm_, i64 0, i32 0, i64 0
  %2 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* %_arg_strm_, i64 0, i32 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(144) %1, i8* nonnull align 8 dereferenceable(144) %2, i64 144, i1 false)
  %GlobalBuf = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 1, i32 1
  %3 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalBuf1, i64 0, i32 0, i32 0, i64 0
  %4 = load i64, i64* %3, align 8
  %5 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalBuf2, i64 0, i32 0, i32 0, i64 0
  %6 = load i64, i64* %5, align 8
  %7 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalBuf3, i64 0, i32 0, i32 0, i64 0
  %8 = load i64, i64* %7, align 8
  %9 = addrspacecast %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* %GlobalBuf to %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*
  %MData.i = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %9, i64 0, i32 1, i32 0
  %arrayidx.i25.i = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %9, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
  store i64 %8, i64 addrspace(4)* %arrayidx.i25.i, align 8, !tbaa !5
  %arrayidx.i23.i = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %9, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  store i64 %4, i64 addrspace(4)* %arrayidx.i23.i, align 8, !tbaa !5
  %arrayidx.i21.i = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %9, i64 0, i32 0, i32 2, i32 0, i32 0, i64 0
  store i64 %6, i64 addrspace(4)* %arrayidx.i21.i, align 8, !tbaa !5
  %add.ptr.i = getelementptr inbounds i8, i8 addrspace(1)* %_arg_GlobalBuf, i64 %8
  store i8 addrspace(1)* %add.ptr.i, i8 addrspace(1)* addrspace(4)* %MData.i, align 8, !tbaa !9
  %GlobalOffset = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 1, i32 2
  %10 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalOffset4, i64 0, i32 0, i32 0, i64 0
  %11 = load i64, i64* %10, align 8
  %12 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalOffset5, i64 0, i32 0, i32 0, i64 0
  %13 = load i64, i64* %12, align 8
  %14 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalOffset6, i64 0, i32 0, i32 0, i64 0
  %15 = load i64, i64* %14, align 8
  %16 = addrspacecast %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* %GlobalOffset to %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*
  %MData.i30 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %16, i64 0, i32 1, i32 0
  %arrayidx.i25.i32 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %16, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
  store i64 %15, i64 addrspace(4)* %arrayidx.i25.i32, align 8, !tbaa !5
  %arrayidx.i23.i34 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %16, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  store i64 %11, i64 addrspace(4)* %arrayidx.i23.i34, align 8, !tbaa !5
  %arrayidx.i21.i36 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %16, i64 0, i32 0, i32 2, i32 0, i32 0, i64 0
  store i64 %13, i64 addrspace(4)* %arrayidx.i21.i36, align 8, !tbaa !5
  %add.ptr.i37 = getelementptr inbounds i32, i32 addrspace(1)* %_arg_GlobalOffset, i64 %15
  store i32 addrspace(1)* %add.ptr.i37, i32 addrspace(1)* addrspace(4)* %MData.i30, align 8, !tbaa !9
  %GlobalFlushBuf = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 1, i32 3
  %17 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalFlushBuf7, i64 0, i32 0, i32 0, i64 0
  %18 = load i64, i64* %17, align 8
  %19 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalFlushBuf8, i64 0, i32 0, i32 0, i64 0
  %20 = load i64, i64* %19, align 8
  %21 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalFlushBuf9, i64 0, i32 0, i32 0, i64 0
  %22 = load i64, i64* %21, align 8
  %23 = addrspacecast %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* %GlobalFlushBuf to %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*
  %MData.i41 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %23, i64 0, i32 1, i32 0
  %arrayidx.i25.i43 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %23, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
  store i64 %22, i64 addrspace(4)* %arrayidx.i25.i43, align 8, !tbaa !5
  %arrayidx.i23.i45 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %23, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  store i64 %18, i64 addrspace(4)* %arrayidx.i23.i45, align 8, !tbaa !5
  %arrayidx.i21.i47 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %23, i64 0, i32 0, i32 2, i32 0, i32 0, i64 0
  store i64 %20, i64 addrspace(4)* %arrayidx.i21.i47, align 8, !tbaa !5
  %add.ptr.i48 = getelementptr inbounds i8, i8 addrspace(1)* %_arg_GlobalFlushBuf, i64 %22
  store i8 addrspace(1)* %add.ptr.i48, i8 addrspace(1)* addrspace(4)* %MData.i41, align 8, !tbaa !9
  %24 = addrspacecast %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* %strm_ to %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)*
  %add.ptr.i.i = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr.i37, i64 1
  %FlushBufferSize.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %24, i64 0, i32 6
  %25 = load i64, i64 addrspace(4)* %FlushBufferSize.i, align 8, !tbaa !10
  %conv.i = trunc i64 %25 to i32
  %call2.i.i = tail call spir_func i32 @_Z18__spirv_AtomicIAddPU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEj(i32 addrspace(1)* %add.ptr.i.i, i32 1, i32 0, i32 %conv.i) #9
  %WIOffset.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %24, i64 0, i32 4
  store i32 %call2.i.i, i32 addrspace(4)* %WIOffset.i, align 8, !tbaa !19
  %conv1.i.i = zext i32 %call2.i.i to i64
  %ptridx.i14.i.i = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr.i48, i64 %conv1.i.i
  %ptridx.ascast.i15.i.i = addrspacecast i8 addrspace(1)* %ptridx.i14.i.i to i8 addrspace(4)*
  store i8 0, i8 addrspace(4)* %ptridx.ascast.i15.i.i, align 1, !tbaa !9
  %add.i.i = add i32 %call2.i.i, 1
  %conv5.i.i = zext i32 %add.i.i to i64
  %ptridx.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr.i48, i64 %conv5.i.i
  %ptridx.ascast.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i.i to i8 addrspace(4)*
  store i8 0, i8 addrspace(4)* %ptridx.ascast.i.i.i, align 1, !tbaa !9
  %26 = addrspacecast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %agg.tmp22 to %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)*
  %27 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !20
  %28 = extractelement <3 x i64> %27, i64 0
  %arrayinit.begin.i.i.i.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %26, i64 0, i32 0, i32 0, i64 0
  store i64 %28, i64 addrspace(4)* %arrayinit.begin.i.i.i.i.i, align 8, !tbaa !5, !alias.scope !20
  %29 = addrspacecast %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t to %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)*
  call spir_func void @_ZNK4test8kernel_tIfEclEN2cl4sycl2idILi1EEE(%"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* dereferenceable_or_null(152) %29, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* nonnull byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %agg.tmp22) #9
  call spir_func void @_ZN2cl4sycl6stream10__finalizeEv(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* dereferenceable_or_null(144) %24) #9
  call void @llvm.lifetime.end.p0i8(i64 152, i8* nonnull %0) #8
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_AtomicIAddPU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEj(i32 addrspace(1)*, i32, i32, i32) local_unnamed_addr #2

; Function Attrs: convergent norecurse
define linkonce_odr dso_local spir_func void @_ZNK4test8kernel_tIfEclEN2cl4sycl2idILi1EEE(%"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* dereferenceable_or_null(152) %this, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %i) local_unnamed_addr #3 comdat align 2 {
entry:
  %ref.tmp = alloca %"struct._ZTSN4test5pod_tE.test::pod_t", align 4
  %strm_ = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.cond.i, %entry
  %Len.0.i = phi i32 [ 0, %entry ], [ %inc.i, %for.cond.i ]
  %idxprom.i = zext i32 %Len.0.i to i64
  %ptridx.i62 = getelementptr inbounds [11 x i8], [11 x i8] addrspace(1)* @.str, i64 0, i64 %idxprom.i
  %ptridx.i = addrspacecast i8 addrspace(1)* %ptridx.i62 to i8 addrspace(4)*
  %0 = load i8, i8 addrspace(4)* %ptridx.i, align 1, !tbaa !9
  %cmp.not.i = icmp eq i8 %0, 0
  %inc.i = add i32 %Len.0.i, 1
  br i1 %cmp.not.i, label %for.end.i, label %for.cond.i, !llvm.loop !27

for.end.i:                                        ; preds = %for.cond.i
  %FlushBufferSize.i = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1, i32 6
  %1 = load i64, i64 addrspace(4)* %FlushBufferSize.i, align 8, !tbaa !10
  %WIOffset.i = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1, i32 4
  %2 = load i32, i32 addrspace(4)* %WIOffset.i, align 8, !tbaa !19
  %conv.i.i.i = zext i32 %2 to i64
  %MData.i.i12.i.i.i = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1, i32 3, i32 1, i32 0
  %3 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i, align 8, !tbaa !9
  %ptridx.i13.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %conv.i.i.i
  %ptridx.ascast.i14.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i13.i.i.i to i8 addrspace(4)*
  %4 = load i8, i8 addrspace(4)* %ptridx.ascast.i14.i.i.i, align 1, !tbaa !9
  %conv1.i.i.i = zext i8 %4 to i32
  %shl.i.i.i = shl nuw nsw i32 %conv1.i.i.i, 8
  %add.i.i.i = add i32 %2, 1
  %conv3.i.i.i = zext i32 %add.i.i.i to i64
  %ptridx.i.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %conv3.i.i.i
  %ptridx.ascast.i.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i.i.i to i8 addrspace(4)*
  %5 = load i8, i8 addrspace(4)* %ptridx.ascast.i.i.i.i, align 1, !tbaa !9
  %conv5.i.i.i = zext i8 %5 to i32
  %add6.i.i.i = or i32 %shl.i.i.i, %conv5.i.i.i
  %add.i.i = add nuw nsw i32 %add6.i.i.i, 2
  %add2.i.i = add i32 %add.i.i, %Len.0.i
  %conv.i.i = zext i32 %add2.i.i to i64
  %cmp.i.i = icmp ult i64 %1, %conv.i.i
  br i1 %cmp.i.i, label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit, label %lor.lhs.false.i.i

lor.lhs.false.i.i:                                ; preds = %for.end.i
  %add5.i.i = add i32 %add2.i.i, %2
  %conv6.i.i = zext i32 %add5.i.i to i64
  %arrayidx.i.i.i.i.i = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1, i32 3, i32 0, i32 1, i32 0, i32 0, i64 0
  %6 = load i64, i64 addrspace(4)* %arrayidx.i.i.i.i.i, align 8, !tbaa !5
  %cmp8.i.i = icmp ult i64 %6, %conv6.i.i
  br i1 %cmp8.i.i, label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit, label %for.cond.preheader.i.i

for.cond.preheader.i.i:                           ; preds = %lor.lhs.false.i.i
  %cmp1868.not.i.i = icmp eq i32 %Len.0.i, 0
  br i1 %cmp1868.not.i.i, label %for.cond.cleanup19.i.i, label %for.body20.i.i.preheader

for.body20.i.i.preheader:                         ; preds = %for.cond.preheader.i.i
  %7 = load i8, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(1)* @.str, i64 0, i64 0) to i8 addrspace(4)*), align 1, !tbaa !9
  %add22.i.i74 = add i32 %add.i.i, %2
  %conv23.i.i75 = zext i32 %add22.i.i74 to i64
  %ptridx.i.i.i76 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %conv23.i.i75
  %ptridx.ascast.i.i.i77 = addrspacecast i8 addrspace(1)* %ptridx.i.i.i76 to i8 addrspace(4)*
  store i8 %7, i8 addrspace(4)* %ptridx.ascast.i.i.i77, align 1, !tbaa !9
  %inc27.i.i78 = add nuw nsw i32 %add6.i.i.i, 3
  %cmp18.i.i79.not = icmp eq i32 %Len.0.i, 1
  br i1 %cmp18.i.i79.not, label %for.cond.cleanup19.i.loopexit.i, label %for.body20.i.for.body20.i_crit_edge.i, !llvm.loop !29

for.cond.cleanup19.i.loopexit.i:                  ; preds = %for.body20.i.for.body20.i_crit_edge.i, %for.body20.i.i.preheader
  %inc27.i.i.lcssa = phi i32 [ %inc27.i.i78, %for.body20.i.i.preheader ], [ %inc27.i.i, %for.body20.i.for.body20.i_crit_edge.i ]
  %.pre8.i = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i, align 8, !tbaa !9
  br label %for.cond.cleanup19.i.i

for.cond.cleanup19.i.i:                           ; preds = %for.cond.cleanup19.i.loopexit.i, %for.cond.preheader.i.i
  %8 = phi i8 addrspace(1)* [ %3, %for.cond.preheader.i.i ], [ %.pre8.i, %for.cond.cleanup19.i.loopexit.i ]
  %Offset.1.lcssa.i.i = phi i32 [ %add.i.i, %for.cond.preheader.i.i ], [ %inc27.i.i.lcssa, %for.cond.cleanup19.i.loopexit.i ]
  %sub.i.i = add i32 %Offset.1.lcssa.i.i, -2
  %shr.i.i.i = lshr i32 %sub.i.i, 8
  %conv.i54.i.i = trunc i32 %shr.i.i.i to i8
  %ptridx.i14.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %8, i64 %conv.i.i.i
  %ptridx.ascast.i15.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i14.i.i.i to i8 addrspace(4)*
  store i8 %conv.i54.i.i, i8 addrspace(4)* %ptridx.ascast.i15.i.i.i, align 1, !tbaa !9
  %conv3.i56.i.i = trunc i32 %sub.i.i to i8
  %9 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i, align 8, !tbaa !9
  %ptridx.i.i59.i.i = getelementptr inbounds i8, i8 addrspace(1)* %9, i64 %conv3.i.i.i
  %ptridx.ascast.i.i60.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i59.i.i to i8 addrspace(4)*
  store i8 %conv3.i56.i.i, i8 addrspace(4)* %ptridx.ascast.i.i60.i.i, align 1, !tbaa !9
  br label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit

for.body20.i.for.body20.i_crit_edge.i:            ; preds = %for.body20.i.i.preheader, %for.body20.i.for.body20.i_crit_edge.i
  %inc27.i.i81 = phi i32 [ %inc27.i.i, %for.body20.i.for.body20.i_crit_edge.i ], [ %inc27.i.i78, %for.body20.i.i.preheader ]
  %inc26.i.i80 = phi i64 [ %inc26.i.i, %for.body20.i.for.body20.i_crit_edge.i ], [ 1, %for.body20.i.i.preheader ]
  %.pre.i = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i, align 8, !tbaa !9
  %ptridx.i.i63 = getelementptr inbounds [11 x i8], [11 x i8] addrspace(1)* @.str, i64 0, i64 %inc26.i.i80
  %ptridx.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i63 to i8 addrspace(4)*
  %10 = load i8, i8 addrspace(4)* %ptridx.i.i, align 1, !tbaa !9
  %add22.i.i = add i32 %inc27.i.i81, %2
  %conv23.i.i = zext i32 %add22.i.i to i64
  %ptridx.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %.pre.i, i64 %conv23.i.i
  %ptridx.ascast.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i.i to i8 addrspace(4)*
  store i8 %10, i8 addrspace(4)* %ptridx.ascast.i.i.i, align 1, !tbaa !9
  %inc26.i.i = add nuw i64 %inc26.i.i80, 1
  %inc27.i.i = add i32 %inc27.i.i81, 1
  %cmp18.i.i = icmp ult i64 %inc26.i.i, %idxprom.i
  br i1 %cmp18.i.i, label %for.body20.i.for.body20.i_crit_edge.i, label %for.cond.cleanup19.i.loopexit.i, !llvm.loop !29

_ZN2cl4sycllsERKNS0_6streamEPKc.exit:             ; preds = %for.end.i, %lor.lhs.false.i.i, %for.cond.cleanup19.i.i
  %11 = bitcast %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %11) #8
  %12 = addrspacecast %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp to %"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)*
  call spir_func void @_Z36__sycl_getCompositeSpecConstantValueIN4test5pod_tEET_PKc(%"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)* sret(%"struct._ZTSN4test5pod_tE.test::pod_t") align 4 %12, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([18 x i8], [18 x i8] addrspace(1)* @__builtin_unique_stable_name._ZNK2cl4sycl6ONEAPI12experimental13spec_constantIN4test5pod_tE11sc_kernel_tE3getIS5_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podISA_EE5valueESA_E4typeEv, i64 0, i64 0) to i8 addrspace(4)*)) #9
  %x = getelementptr inbounds %"struct._ZTSN4test5pod_tE.test::pod_t", %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp, i64 0, i32 0
  %13 = addrspacecast float* %x to float addrspace(4)*
  %call2 = call spir_func align 8 dereferenceable(144) %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* @_ZN2cl4sycllsERKNS0_6streamERKf(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* align 8 dereferenceable(144) %strm_, float addrspace(4)* align 4 dereferenceable(4) %13) #9
  br label %for.cond.i9

for.cond.i9:                                      ; preds = %for.cond.i9, %_ZN2cl4sycllsERKNS0_6streamEPKc.exit
  %Len.0.i4 = phi i32 [ 0, %_ZN2cl4sycllsERKNS0_6streamEPKc.exit ], [ %inc.i8, %for.cond.i9 ]
  %idxprom.i5 = zext i32 %Len.0.i4 to i64
  %ptridx.i664 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(1)* @.str.1, i64 0, i64 %idxprom.i5
  %ptridx.i6 = addrspacecast i8 addrspace(1)* %ptridx.i664 to i8 addrspace(4)*
  %14 = load i8, i8 addrspace(4)* %ptridx.i6, align 1, !tbaa !9
  %cmp.not.i7 = icmp eq i8 %14, 0
  %inc.i8 = add i32 %Len.0.i4, 1
  br i1 %cmp.not.i7, label %for.end.i28, label %for.cond.i9, !llvm.loop !27

for.end.i28:                                      ; preds = %for.cond.i9
  %FlushBufferSize.i10 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %call2, i64 0, i32 6
  %15 = load i64, i64 addrspace(4)* %FlushBufferSize.i10, align 8, !tbaa !10
  %WIOffset.i11 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %call2, i64 0, i32 4
  %16 = load i32, i32 addrspace(4)* %WIOffset.i11, align 8, !tbaa !19
  %conv.i.i.i12 = zext i32 %16 to i64
  %MData.i.i12.i.i.i13 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %call2, i64 0, i32 3, i32 1, i32 0
  %17 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i13, align 8, !tbaa !9
  %ptridx.i13.i.i.i14 = getelementptr inbounds i8, i8 addrspace(1)* %17, i64 %conv.i.i.i12
  %ptridx.ascast.i14.i.i.i15 = addrspacecast i8 addrspace(1)* %ptridx.i13.i.i.i14 to i8 addrspace(4)*
  %18 = load i8, i8 addrspace(4)* %ptridx.ascast.i14.i.i.i15, align 1, !tbaa !9
  %conv1.i.i.i16 = zext i8 %18 to i32
  %shl.i.i.i17 = shl nuw nsw i32 %conv1.i.i.i16, 8
  %add.i.i.i18 = add i32 %16, 1
  %conv3.i.i.i19 = zext i32 %add.i.i.i18 to i64
  %ptridx.i.i.i.i20 = getelementptr inbounds i8, i8 addrspace(1)* %17, i64 %conv3.i.i.i19
  %ptridx.ascast.i.i.i.i21 = addrspacecast i8 addrspace(1)* %ptridx.i.i.i.i20 to i8 addrspace(4)*
  %19 = load i8, i8 addrspace(4)* %ptridx.ascast.i.i.i.i21, align 1, !tbaa !9
  %conv5.i.i.i22 = zext i8 %19 to i32
  %add6.i.i.i23 = or i32 %shl.i.i.i17, %conv5.i.i.i22
  %add.i.i24 = add nuw nsw i32 %add6.i.i.i23, 2
  %add2.i.i25 = add i32 %add.i.i24, %Len.0.i4
  %conv.i.i26 = zext i32 %add2.i.i25 to i64
  %cmp.i.i27 = icmp ult i64 %15, %conv.i.i26
  br i1 %cmp.i.i27, label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit61, label %lor.lhs.false.i.i33

lor.lhs.false.i.i33:                              ; preds = %for.end.i28
  %add5.i.i29 = add i32 %add2.i.i25, %16
  %conv6.i.i30 = zext i32 %add5.i.i29 to i64
  %arrayidx.i.i.i.i.i31 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %call2, i64 0, i32 3, i32 0, i32 1, i32 0, i32 0, i64 0
  %20 = load i64, i64 addrspace(4)* %arrayidx.i.i.i.i.i31, align 8, !tbaa !5
  %cmp8.i.i32 = icmp ult i64 %20, %conv6.i.i30
  br i1 %cmp8.i.i32, label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit61, label %for.cond.preheader.i.i35

for.cond.preheader.i.i35:                         ; preds = %lor.lhs.false.i.i33
  %cmp1868.not.i.i34 = icmp eq i32 %Len.0.i4, 0
  br i1 %cmp1868.not.i.i34, label %for.cond.cleanup19.i.i47, label %for.body20.i.i58.preheader

for.body20.i.i58.preheader:                       ; preds = %for.cond.preheader.i.i35
  %21 = load i8, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(1)* @.str.1, i64 0, i64 0) to i8 addrspace(4)*), align 1, !tbaa !9
  %add22.i.i5166 = add i32 %add.i.i24, %16
  %conv23.i.i5267 = zext i32 %add22.i.i5166 to i64
  %ptridx.i.i.i5368 = getelementptr inbounds i8, i8 addrspace(1)* %17, i64 %conv23.i.i5267
  %ptridx.ascast.i.i.i5469 = addrspacecast i8 addrspace(1)* %ptridx.i.i.i5368 to i8 addrspace(4)*
  store i8 %21, i8 addrspace(4)* %ptridx.ascast.i.i.i5469, align 1, !tbaa !9
  %inc27.i.i5670 = add nuw nsw i32 %add6.i.i.i23, 3
  %cmp18.i.i5771.not = icmp eq i32 %Len.0.i4, 1
  br i1 %cmp18.i.i5771.not, label %for.cond.cleanup19.i.loopexit.i37, label %for.body20.i.for.body20.i_crit_edge.i60, !llvm.loop !29

for.cond.cleanup19.i.loopexit.i37:                ; preds = %for.body20.i.for.body20.i_crit_edge.i60, %for.body20.i.i58.preheader
  %inc27.i.i56.lcssa = phi i32 [ %inc27.i.i5670, %for.body20.i.i58.preheader ], [ %inc27.i.i56, %for.body20.i.for.body20.i_crit_edge.i60 ]
  %.pre8.i36 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i13, align 8, !tbaa !9
  br label %for.cond.cleanup19.i.i47

for.cond.cleanup19.i.i47:                         ; preds = %for.cond.cleanup19.i.loopexit.i37, %for.cond.preheader.i.i35
  %22 = phi i8 addrspace(1)* [ %17, %for.cond.preheader.i.i35 ], [ %.pre8.i36, %for.cond.cleanup19.i.loopexit.i37 ]
  %Offset.1.lcssa.i.i38 = phi i32 [ %add.i.i24, %for.cond.preheader.i.i35 ], [ %inc27.i.i56.lcssa, %for.cond.cleanup19.i.loopexit.i37 ]
  %sub.i.i39 = add i32 %Offset.1.lcssa.i.i38, -2
  %shr.i.i.i40 = lshr i32 %sub.i.i39, 8
  %conv.i54.i.i41 = trunc i32 %shr.i.i.i40 to i8
  %ptridx.i14.i.i.i42 = getelementptr inbounds i8, i8 addrspace(1)* %22, i64 %conv.i.i.i12
  %ptridx.ascast.i15.i.i.i43 = addrspacecast i8 addrspace(1)* %ptridx.i14.i.i.i42 to i8 addrspace(4)*
  store i8 %conv.i54.i.i41, i8 addrspace(4)* %ptridx.ascast.i15.i.i.i43, align 1, !tbaa !9
  %conv3.i56.i.i44 = trunc i32 %sub.i.i39 to i8
  %23 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i13, align 8, !tbaa !9
  %ptridx.i.i59.i.i45 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %conv3.i.i.i19
  %ptridx.ascast.i.i60.i.i46 = addrspacecast i8 addrspace(1)* %ptridx.i.i59.i.i45 to i8 addrspace(4)*
  store i8 %conv3.i56.i.i44, i8 addrspace(4)* %ptridx.ascast.i.i60.i.i46, align 1, !tbaa !9
  br label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit61

for.body20.i.for.body20.i_crit_edge.i60:          ; preds = %for.body20.i.i58.preheader, %for.body20.i.for.body20.i_crit_edge.i60
  %inc27.i.i5673 = phi i32 [ %inc27.i.i56, %for.body20.i.for.body20.i_crit_edge.i60 ], [ %inc27.i.i5670, %for.body20.i.i58.preheader ]
  %inc26.i.i5572 = phi i64 [ %inc26.i.i55, %for.body20.i.for.body20.i_crit_edge.i60 ], [ 1, %for.body20.i.i58.preheader ]
  %.pre.i59 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i13, align 8, !tbaa !9
  %ptridx.i.i5065 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(1)* @.str.1, i64 0, i64 %inc26.i.i5572
  %ptridx.i.i50 = addrspacecast i8 addrspace(1)* %ptridx.i.i5065 to i8 addrspace(4)*
  %24 = load i8, i8 addrspace(4)* %ptridx.i.i50, align 1, !tbaa !9
  %add22.i.i51 = add i32 %inc27.i.i5673, %16
  %conv23.i.i52 = zext i32 %add22.i.i51 to i64
  %ptridx.i.i.i53 = getelementptr inbounds i8, i8 addrspace(1)* %.pre.i59, i64 %conv23.i.i52
  %ptridx.ascast.i.i.i54 = addrspacecast i8 addrspace(1)* %ptridx.i.i.i53 to i8 addrspace(4)*
  store i8 %24, i8 addrspace(4)* %ptridx.ascast.i.i.i54, align 1, !tbaa !9
  %inc26.i.i55 = add nuw i64 %inc26.i.i5572, 1
  %inc27.i.i56 = add i32 %inc27.i.i5673, 1
  %cmp18.i.i57 = icmp ult i64 %inc26.i.i55, %idxprom.i5
  br i1 %cmp18.i.i57, label %for.body20.i.for.body20.i_crit_edge.i60, label %for.cond.cleanup19.i.loopexit.i37, !llvm.loop !29

_ZN2cl4sycllsERKNS0_6streamEPKc.exit61:           ; preds = %for.end.i28, %lor.lhs.false.i.i33, %for.cond.cleanup19.i.i47
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %11) #8
  ret void
}

; Function Attrs: convergent norecurse
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6stream10__finalizeEv(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* dereferenceable_or_null(144) %this) local_unnamed_addr #3 comdat align 2 {
entry:
  %WIOffset = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %this, i64 0, i32 4
  %0 = load i32, i32 addrspace(4)* %WIOffset, align 8, !tbaa !19
  %conv.i.i = zext i32 %0 to i64
  %MData.i.i12.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %this, i64 0, i32 3, i32 1, i32 0
  %1 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %ptridx.i13.i.i = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %conv.i.i
  %ptridx.ascast.i14.i.i = addrspacecast i8 addrspace(1)* %ptridx.i13.i.i to i8 addrspace(4)*
  %2 = load i8, i8 addrspace(4)* %ptridx.ascast.i14.i.i, align 1, !tbaa !9
  %conv1.i.i = zext i8 %2 to i32
  %shl.i.i = shl nuw nsw i32 %conv1.i.i, 8
  %add.i.i = add i32 %0, 1
  %conv3.i.i = zext i32 %add.i.i to i64
  %ptridx.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %conv3.i.i
  %ptridx.ascast.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i.i to i8 addrspace(4)*
  %3 = load i8, i8 addrspace(4)* %ptridx.ascast.i.i.i, align 1, !tbaa !9
  %conv5.i.i = zext i8 %3 to i32
  %add6.i.i = or i32 %shl.i.i, %conv5.i.i
  %cmp.i = icmp eq i32 %add6.i.i, 0
  br i1 %cmp.i, label %_ZN2cl4sycl6detail11flushBufferERNS0_8accessorIjLi1ELNS0_6access4modeE1029ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEERNS2_IcLi1ELS4_1026ELS5_2014ELS6_0ES9_EESD_j.exit, label %if.end.i

if.end.i:                                         ; preds = %entry
  %MData.i.i.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %this, i64 0, i32 2, i32 1, i32 0
  %4 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %MData.i.i.i.i, align 8, !tbaa !9, !noalias !30
  %call2.i.i.i = tail call spir_func i32 @_Z18__spirv_AtomicLoadPU3AS1KjN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(i32 addrspace(1)* %4, i32 1, i32 0) #9
  %5 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %this, i64 0, i32 1, i32 0, i32 1, i32 0, i32 0, i64 0
  %conv3.i36.i = zext i32 %add6.i.i to i64
  br label %do.body.i.i

do.body.i.i:                                      ; preds = %if.end.i.i, %if.end.i
  %storemerge.i.i = phi i32 [ %call2.i.i.i, %if.end.i ], [ %call3.i.i.i, %if.end.i.i ]
  %6 = load i64, i64 addrspace(4)* %5, align 8, !noalias !33
  %conv.i37.i = zext i32 %storemerge.i.i to i64
  %sub.i.i = sub i64 %6, %conv.i37.i
  %cmp.i.i = icmp ult i64 %sub.i.i, %conv3.i36.i
  br i1 %cmp.i.i, label %_ZN2cl4sycl6detail11flushBufferERNS0_8accessorIjLi1ELNS0_6access4modeE1029ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEERNS2_IcLi1ELS4_1026ELS5_2014ELS6_0ES9_EESD_j.exit, label %if.end.i.i

if.end.i.i:                                       ; preds = %do.body.i.i
  %add.i38.i = add i32 %storemerge.i.i, %add6.i.i
  %7 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %MData.i.i.i.i, align 8, !tbaa !9, !noalias !36
  %call3.i.i.i = tail call spir_func i32 @_Z29__spirv_AtomicCompareExchangePU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_jj(i32 addrspace(1)* %7, i32 1, i32 0, i32 0, i32 %add.i38.i, i32 %storemerge.i.i) #9
  %cmp.i.i.i = icmp eq i32 %call3.i.i.i, %storemerge.i.i
  br i1 %cmp.i.i.i, label %if.end3.i, label %do.body.i.i

if.end3.i:                                        ; preds = %if.end.i.i
  %add.i = add i32 %0, 2
  %add4.i = add i32 %add6.i.i, %add.i
  %cmp544.i = icmp ult i32 %add.i, %add4.i
  br i1 %cmp544.i, label %for.body.lr.ph.i, label %for.cond.cleanup.i

for.body.lr.ph.i:                                 ; preds = %if.end3.i
  %MData.i.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %this, i64 0, i32 1, i32 1, i32 0
  br label %for.body.i

for.cond.cleanup.i:                               ; preds = %for.body.i, %if.end3.i
  %8 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %ptridx.i14.i.i = getelementptr inbounds i8, i8 addrspace(1)* %8, i64 %conv.i.i
  %ptridx.ascast.i15.i.i = addrspacecast i8 addrspace(1)* %ptridx.i14.i.i to i8 addrspace(4)*
  store i8 0, i8 addrspace(4)* %ptridx.ascast.i15.i.i, align 1, !tbaa !9
  %9 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %ptridx.i.i34.i = getelementptr inbounds i8, i8 addrspace(1)* %9, i64 %conv3.i.i
  %ptridx.ascast.i.i35.i = addrspacecast i8 addrspace(1)* %ptridx.i.i34.i to i8 addrspace(4)*
  store i8 0, i8 addrspace(4)* %ptridx.ascast.i.i35.i, align 1, !tbaa !9
  br label %_ZN2cl4sycl6detail11flushBufferERNS0_8accessorIjLi1ELNS0_6access4modeE1029ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEERNS2_IcLi1ELS4_1026ELS5_2014ELS6_0ES9_EESD_j.exit

for.body.i:                                       ; preds = %for.body.i, %for.body.lr.ph.i
  %I.046.i = phi i32 [ %add.i, %for.body.lr.ph.i ], [ %inc10.i, %for.body.i ]
  %Cur.045.i = phi i32 [ %storemerge.i.i, %for.body.lr.ph.i ], [ %inc.i, %for.body.i ]
  %conv.i = zext i32 %I.046.i to i64
  %10 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %ptridx.i28.i = getelementptr inbounds i8, i8 addrspace(1)* %10, i64 %conv.i
  %ptridx.ascast.i29.i = addrspacecast i8 addrspace(1)* %ptridx.i28.i to i8 addrspace(4)*
  %11 = load i8, i8 addrspace(4)* %ptridx.ascast.i29.i, align 1, !tbaa !9
  %inc.i = add i32 %Cur.045.i, 1
  %conv8.i = zext i32 %Cur.045.i to i64
  %12 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i.i, align 8, !tbaa !9
  %ptridx.i.i = getelementptr inbounds i8, i8 addrspace(1)* %12, i64 %conv8.i
  %ptridx.ascast.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i to i8 addrspace(4)*
  store i8 %11, i8 addrspace(4)* %ptridx.ascast.i.i, align 1, !tbaa !9
  %inc10.i = add nuw i32 %I.046.i, 1
  %cmp5.i = icmp ult i32 %inc10.i, %add4.i
  br i1 %cmp5.i, label %for.body.i, label %for.cond.cleanup.i, !llvm.loop !39

_ZN2cl4sycl6detail11flushBufferERNS0_8accessorIjLi1ELNS0_6access4modeE1029ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEERNS2_IcLi1ELS4_1026ELS5_2014ELS6_0ES9_EESD_j.exit: ; preds = %do.body.i.i, %entry, %for.cond.cleanup.i
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_AtomicLoadPU3AS1KjN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE(i32 addrspace(1)*, i32, i32) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z29__spirv_AtomicCompareExchangePU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagES5_jj(i32 addrspace(1)*, i32, i32, i32, i32, i32) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func void @_Z36__sycl_getCompositeSpecConstantValueIN4test5pod_tEET_PKc(%"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)* sret(%"struct._ZTSN4test5pod_tE.test::pod_t") align 4, i8 addrspace(4)*) local_unnamed_addr #2

; Function Attrs: convergent inlinehint norecurse
define linkonce_odr dso_local spir_func align 8 dereferenceable(144) %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* @_ZN2cl4sycllsERKNS0_6streamERKf(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* align 8 dereferenceable(144) %Out, float addrspace(4)* align 4 dereferenceable(4) %RHS) local_unnamed_addr #4 comdat {
entry:
  %Digits.i = alloca [24 x i8], align 1
  %FlushBufferSize = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %Out, i64 0, i32 6
  %0 = load i64, i64 addrspace(4)* %FlushBufferSize, align 8, !tbaa !10
  %WIOffset = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %Out, i64 0, i32 4
  %1 = load i32, i32 addrspace(4)* %WIOffset, align 8, !tbaa !19
  %Flags.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %Out, i64 0, i32 10
  %2 = load i32, i32 addrspace(4)* %Flags.i, align 4, !tbaa !40
  %Width.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %Out, i64 0, i32 9
  %3 = load i32, i32 addrspace(4)* %Width.i, align 8, !tbaa !41
  %Precision.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %Out, i64 0, i32 8
  %4 = load i32, i32 addrspace(4)* %Precision.i, align 4, !tbaa !42
  %5 = getelementptr inbounds [24 x i8], [24 x i8]* %Digits.i, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %5) #8
  call void @llvm.memset.p0i8.i64(i8* nonnull align 1 dereferenceable(24) %5, i8 0, i64 24, i1 false)
  %6 = addrspacecast i8* %5 to i8 addrspace(4)*
  %7 = load float, float addrspace(4)* %RHS, align 4, !tbaa !43
  %call.i.i = call spir_func i32 @_ZN2cl4sycl6detail14checkForInfNanIfEENSt9enable_ifIXoosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valueEjE4typeEPcS4_(i8 addrspace(4)* %6, float %7) #9
  %tobool.not.i.i = icmp eq i32 %call.i.i, 0
  br i1 %tobool.not.i.i, label %if.end.i.i, label %_ZN2cl4sycl6detail18writeFloatingPointIfEENSt9enable_ifIXoooosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valuesr3std7is_sameIS4_NS1_9half_impl4halfEEE5valueEvE4typeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNSA_6targetE2014ELNSA_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjjiiRKS4_.exit

if.end.i.i:                                       ; preds = %entry
  %8 = load float, float addrspace(4)* %RHS, align 4, !tbaa !43
  %cmp.i.i = fcmp olt float %8, 0.000000e+00
  %fneg.i.i = fneg float %8
  %cond.i.i = select i1 %cmp.i.i, float %fneg.i.i, float %8
  br i1 %cmp.i.i, label %if.then2.i.i, label %if.else.i.i

if.then2.i.i:                                     ; preds = %if.end.i.i
  store i8 45, i8 addrspace(4)* %6, align 1, !tbaa !9
  br label %if.end9.i.i

if.else.i.i:                                      ; preds = %if.end.i.i
  %and.i.i = and i32 %2, 16
  %tobool3.not.i.i = icmp eq i32 %and.i.i, 0
  br i1 %tobool3.not.i.i, label %if.end9.i.i, label %if.then4.i.i

if.then4.i.i:                                     ; preds = %if.else.i.i
  store i8 43, i8 addrspace(4)* %6, align 1, !tbaa !9
  br label %if.end9.i.i

if.end9.i.i:                                      ; preds = %if.then4.i.i, %if.else.i.i, %if.then2.i.i
  %Offset.0.i.i = phi i32 [ 1, %if.then2.i.i ], [ 1, %if.then4.i.i ], [ 0, %if.else.i.i ]
  %and10.i.i = and i32 %2, 64
  %tobool11.not.i.i = icmp ne i32 %and10.i.i, 0
  %idx.ext.i.i = zext i32 %Offset.0.i.i to i64
  %add.ptr.i9.i = getelementptr inbounds [24 x i8], [24 x i8]* %Digits.i, i64 0, i64 %idx.ext.i.i
  %add.ptr.i.i = addrspacecast i8* %add.ptr.i9.i to i8 addrspace(4)*
  %call15.i.i = call spir_func i32 @_ZN2cl4sycl6detail21floatingPointToDecStrIfEENSt9enable_ifIXoooosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valuesr3std7is_sameIS4_NS1_9half_impl4halfEEE5valueEjE4typeES4_Pcib(float %cond.i.i, i8 addrspace(4)* %add.ptr.i.i, i32 %4, i1 zeroext %tobool11.not.i.i) #9
  %add.i.i = add i32 %call15.i.i, %Offset.0.i.i
  br label %_ZN2cl4sycl6detail18writeFloatingPointIfEENSt9enable_ifIXoooosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valuesr3std7is_sameIS4_NS1_9half_impl4halfEEE5valueEvE4typeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNSA_6targetE2014ELNSA_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjjiiRKS4_.exit

_ZN2cl4sycl6detail18writeFloatingPointIfEENSt9enable_ifIXoooosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valuesr3std7is_sameIS4_NS1_9half_impl4halfEEE5valueEvE4typeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNSA_6targetE2014ELNSA_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjjiiRKS4_.exit: ; preds = %entry, %if.end9.i.i
  %retval.0.i.i = phi i32 [ %add.i.i, %if.end9.i.i ], [ %call.i.i, %entry ]
  %cmp.i = icmp sgt i32 %3, 0
  %cmp2.i = icmp ugt i32 %3, %retval.0.i.i
  %or.cond.i = and i1 %cmp.i, %cmp2.i
  %sub.i = sub i32 %3, %retval.0.i.i
  %cond.i = select i1 %or.cond.i, i32 %sub.i, i32 0
  %conv.i.i = zext i32 %1 to i64
  %MData.i.i12.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %Out, i64 0, i32 3, i32 1, i32 0
  %9 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %ptridx.i13.i.i = getelementptr inbounds i8, i8 addrspace(1)* %9, i64 %conv.i.i
  %ptridx.ascast.i14.i.i = addrspacecast i8 addrspace(1)* %ptridx.i13.i.i to i8 addrspace(4)*
  %10 = load i8, i8 addrspace(4)* %ptridx.ascast.i14.i.i, align 1, !tbaa !9
  %conv1.i.i = zext i8 %10 to i32
  %shl.i.i = shl nuw nsw i32 %conv1.i.i, 8
  %add.i.i10 = add i32 %1, 1
  %conv3.i.i = zext i32 %add.i.i10 to i64
  %ptridx.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %9, i64 %conv3.i.i
  %ptridx.ascast.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i.i to i8 addrspace(4)*
  %11 = load i8, i8 addrspace(4)* %ptridx.ascast.i.i.i, align 1, !tbaa !9
  %conv5.i.i = zext i8 %11 to i32
  %add6.i.i = or i32 %shl.i.i, %conv5.i.i
  %add.i = add nuw nsw i32 %add6.i.i, 2
  %add1.i = add i32 %cond.i, %retval.0.i.i
  %add2.i = add i32 %add.i, %add1.i
  %conv.i11 = zext i32 %add2.i to i64
  %cmp.i12 = icmp ult i64 %0, %conv.i11
  br i1 %cmp.i12, label %_ZN2cl4sycl6detail5writeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjPKcjj.exit, label %lor.lhs.false.i

lor.lhs.false.i:                                  ; preds = %_ZN2cl4sycl6detail18writeFloatingPointIfEENSt9enable_ifIXoooosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valuesr3std7is_sameIS4_NS1_9half_impl4halfEEE5valueEvE4typeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNSA_6targetE2014ELNSA_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjjiiRKS4_.exit
  %add5.i = add i32 %add2.i, %1
  %conv6.i = zext i32 %add5.i to i64
  %arrayidx.i.i.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %Out, i64 0, i32 3, i32 0, i32 1, i32 0, i32 0, i64 0
  %12 = load i64, i64 addrspace(4)* %arrayidx.i.i.i.i, align 8, !tbaa !5
  %cmp8.i = icmp ult i64 %12, %conv6.i
  br i1 %cmp8.i, label %_ZN2cl4sycl6detail5writeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjPKcjj.exit, label %for.cond.preheader.i

for.cond.preheader.i:                             ; preds = %lor.lhs.false.i
  %conv9.i = zext i32 %cond.i to i64
  %cmp1071.not.i = icmp eq i32 %cond.i, 0
  br i1 %cmp1071.not.i, label %for.cond16.preheader.i, label %for.body.i.preheader

for.body.i.preheader:                             ; preds = %for.cond.preheader.i
  %add11.i16 = add i32 %add.i, %1
  %conv12.i17 = zext i32 %add11.i16 to i64
  %ptridx.i63.i18 = getelementptr inbounds i8, i8 addrspace(1)* %9, i64 %conv12.i17
  %ptridx.ascast.i64.i19 = addrspacecast i8 addrspace(1)* %ptridx.i63.i18 to i8 addrspace(4)*
  store i8 32, i8 addrspace(4)* %ptridx.ascast.i64.i19, align 1, !tbaa !9
  %inc14.i20 = add nuw nsw i32 %add6.i.i, 3
  %cmp10.i21.not = icmp eq i32 %cond.i, 1
  br i1 %cmp10.i21.not, label %for.cond16.preheader.i, label %for.body.for.body_crit_edge.i, !llvm.loop !45

for.cond16.preheader.i:                           ; preds = %for.body.for.body_crit_edge.i, %for.body.i.preheader, %for.cond.preheader.i
  %Offset.0.lcssa.i = phi i32 [ %add.i, %for.cond.preheader.i ], [ %inc14.i20, %for.body.i.preheader ], [ %inc14.i, %for.body.for.body_crit_edge.i ]
  %conv17.i = zext i32 %retval.0.i.i to i64
  %cmp1868.not.i = icmp eq i32 %retval.0.i.i, 0
  br i1 %cmp1868.not.i, label %for.cond.cleanup19.i, label %for.body20.i

for.body.for.body_crit_edge.i:                    ; preds = %for.body.i.preheader, %for.body.for.body_crit_edge.i
  %inc14.i23 = phi i32 [ %inc14.i, %for.body.for.body_crit_edge.i ], [ %inc14.i20, %for.body.i.preheader ]
  %inc.i22 = phi i64 [ %inc.i, %for.body.for.body_crit_edge.i ], [ 1, %for.body.i.preheader ]
  %.pre.i = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %add11.i = add i32 %inc14.i23, %1
  %conv12.i = zext i32 %add11.i to i64
  %ptridx.i63.i = getelementptr inbounds i8, i8 addrspace(1)* %.pre.i, i64 %conv12.i
  %ptridx.ascast.i64.i = addrspacecast i8 addrspace(1)* %ptridx.i63.i to i8 addrspace(4)*
  store i8 32, i8 addrspace(4)* %ptridx.ascast.i64.i, align 1, !tbaa !9
  %inc.i = add nuw i64 %inc.i22, 1
  %inc14.i = add i32 %inc14.i23, 1
  %cmp10.i = icmp ult i64 %inc.i, %conv9.i
  br i1 %cmp10.i, label %for.body.for.body_crit_edge.i, label %for.cond16.preheader.i, !llvm.loop !45

for.cond.cleanup19.i:                             ; preds = %for.body20.i, %for.cond16.preheader.i
  %Offset.1.lcssa.i = phi i32 [ %Offset.0.lcssa.i, %for.cond16.preheader.i ], [ %inc27.i, %for.body20.i ]
  %sub.i13 = add i32 %Offset.1.lcssa.i, -2
  %shr.i.i = lshr i32 %sub.i13, 8
  %conv.i54.i = trunc i32 %shr.i.i to i8
  %13 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %ptridx.i14.i.i = getelementptr inbounds i8, i8 addrspace(1)* %13, i64 %conv.i.i
  %ptridx.ascast.i15.i.i = addrspacecast i8 addrspace(1)* %ptridx.i14.i.i to i8 addrspace(4)*
  store i8 %conv.i54.i, i8 addrspace(4)* %ptridx.ascast.i15.i.i, align 1, !tbaa !9
  %conv3.i56.i = trunc i32 %sub.i13 to i8
  %14 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %ptridx.i.i59.i = getelementptr inbounds i8, i8 addrspace(1)* %14, i64 %conv3.i.i
  %ptridx.ascast.i.i60.i = addrspacecast i8 addrspace(1)* %ptridx.i.i59.i to i8 addrspace(4)*
  store i8 %conv3.i56.i, i8 addrspace(4)* %ptridx.ascast.i.i60.i, align 1, !tbaa !9
  br label %_ZN2cl4sycl6detail5writeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjPKcjj.exit

for.body20.i:                                     ; preds = %for.cond16.preheader.i, %for.body20.i
  %I15.070.i = phi i64 [ %inc26.i, %for.body20.i ], [ 0, %for.cond16.preheader.i ]
  %Offset.169.i = phi i32 [ %inc27.i, %for.body20.i ], [ %Offset.0.lcssa.i, %for.cond16.preheader.i ]
  %ptridx.i15 = getelementptr inbounds [24 x i8], [24 x i8]* %Digits.i, i64 0, i64 %I15.070.i
  %ptridx.i = addrspacecast i8* %ptridx.i15 to i8 addrspace(4)*
  %15 = load i8, i8 addrspace(4)* %ptridx.i, align 1, !tbaa !9
  %add22.i = add i32 %Offset.169.i, %1
  %conv23.i = zext i32 %add22.i to i64
  %16 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i, align 8, !tbaa !9
  %ptridx.i.i = getelementptr inbounds i8, i8 addrspace(1)* %16, i64 %conv23.i
  %ptridx.ascast.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i to i8 addrspace(4)*
  store i8 %15, i8 addrspace(4)* %ptridx.ascast.i.i, align 1, !tbaa !9
  %inc26.i = add nuw i64 %I15.070.i, 1
  %inc27.i = add i32 %Offset.169.i, 1
  %cmp18.i = icmp ult i64 %inc26.i, %conv17.i
  br i1 %cmp18.i, label %for.body20.i, label %for.cond.cleanup19.i, !llvm.loop !29

_ZN2cl4sycl6detail5writeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjPKcjj.exit: ; preds = %_ZN2cl4sycl6detail18writeFloatingPointIfEENSt9enable_ifIXoooosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valuesr3std7is_sameIS4_NS1_9half_impl4halfEEE5valueEvE4typeERNS0_8accessorIcLi1ELNS0_6access4modeE1026ELNSA_6targetE2014ELNSA_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEEmjjiiRKS4_.exit, %lor.lhs.false.i, %for.cond.cleanup19.i
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %5) #8
  ret %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %Out
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #5

; Function Attrs: convergent inlinehint norecurse
define linkonce_odr dso_local spir_func i32 @_ZN2cl4sycl6detail14checkForInfNanIfEENSt9enable_ifIXoosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valueEjE4typeEPcS4_(i8 addrspace(4)* %Buf, float %Val) local_unnamed_addr #4 comdat {
entry:
  %call1.i.i = tail call zeroext i1 @_Z13__spirv_IsNanf(float %Val) #10
  br i1 %call1.i.i, label %for.cond.i, label %if.end

for.cond.i:                                       ; preds = %entry, %for.cond.i
  %Len.0.i = phi i32 [ %inc.i, %for.cond.i ], [ 0, %entry ]
  %idxprom.i = zext i32 %Len.0.i to i64
  %ptridx.i50 = getelementptr inbounds [4 x i8], [4 x i8] addrspace(1)* @.str.2, i64 0, i64 %idxprom.i
  %ptridx.i = addrspacecast i8 addrspace(1)* %ptridx.i50 to i8 addrspace(4)*
  %0 = load i8, i8 addrspace(4)* %ptridx.i, align 1, !tbaa !9
  %cmp.not.i = icmp eq i8 %0, 0
  %inc.i = add i32 %Len.0.i, 1
  br i1 %cmp.not.i, label %for.cond1.preheader.i, label %for.cond.i, !llvm.loop !46

for.cond1.preheader.i:                            ; preds = %for.cond.i
  %cmp220.not.i = icmp eq i32 %Len.0.i, 0
  br i1 %cmp220.not.i, label %return, label %for.body3.i

for.body3.i:                                      ; preds = %for.cond1.preheader.i, %for.body3.i
  %I.021.i = phi i32 [ %inc9.i, %for.body3.i ], [ 0, %for.cond1.preheader.i ]
  %idxprom4.i = zext i32 %I.021.i to i64
  %ptridx5.i51 = getelementptr inbounds [4 x i8], [4 x i8] addrspace(1)* @.str.2, i64 0, i64 %idxprom4.i
  %ptridx5.i = addrspacecast i8 addrspace(1)* %ptridx5.i51 to i8 addrspace(4)*
  %1 = load i8, i8 addrspace(4)* %ptridx5.i, align 1, !tbaa !9
  %ptridx7.i = getelementptr inbounds i8, i8 addrspace(4)* %Buf, i64 %idxprom4.i
  store i8 %1, i8 addrspace(4)* %ptridx7.i, align 1, !tbaa !9
  %inc9.i = add nuw i32 %I.021.i, 1
  %cmp2.i = icmp ult i32 %inc9.i, %Len.0.i
  br i1 %cmp2.i, label %for.body3.i, label %return, !llvm.loop !47

if.end:                                           ; preds = %entry
  %call1.i.i49 = tail call zeroext i1 @_Z13__spirv_IsInff(float %Val) #10
  br i1 %call1.i.i49, label %if.then4, label %return

if.then4:                                         ; preds = %if.end
  %call1.i.i48 = tail call zeroext i1 @_Z18__spirv_SignBitSetf(float %Val) #10
  br i1 %call1.i.i48, label %for.cond.i37, label %for.cond.i21

for.cond.i37:                                     ; preds = %if.then4, %for.cond.i37
  %Len.0.i32 = phi i32 [ %inc.i36, %for.cond.i37 ], [ 0, %if.then4 ]
  %idxprom.i33 = zext i32 %Len.0.i32 to i64
  %ptridx.i3452 = getelementptr inbounds [5 x i8], [5 x i8] addrspace(1)* @.str.3, i64 0, i64 %idxprom.i33
  %ptridx.i34 = addrspacecast i8 addrspace(1)* %ptridx.i3452 to i8 addrspace(4)*
  %2 = load i8, i8 addrspace(4)* %ptridx.i34, align 1, !tbaa !9
  %cmp.not.i35 = icmp eq i8 %2, 0
  %inc.i36 = add i32 %Len.0.i32, 1
  br i1 %cmp.not.i35, label %for.cond1.preheader.i39, label %for.cond.i37, !llvm.loop !46

for.cond1.preheader.i39:                          ; preds = %for.cond.i37
  %cmp220.not.i38 = icmp eq i32 %Len.0.i32, 0
  br i1 %cmp220.not.i38, label %return, label %for.body3.i46

for.body3.i46:                                    ; preds = %for.cond1.preheader.i39, %for.body3.i46
  %I.021.i40 = phi i32 [ %inc9.i44, %for.body3.i46 ], [ 0, %for.cond1.preheader.i39 ]
  %idxprom4.i41 = zext i32 %I.021.i40 to i64
  %ptridx5.i4253 = getelementptr inbounds [5 x i8], [5 x i8] addrspace(1)* @.str.3, i64 0, i64 %idxprom4.i41
  %ptridx5.i42 = addrspacecast i8 addrspace(1)* %ptridx5.i4253 to i8 addrspace(4)*
  %3 = load i8, i8 addrspace(4)* %ptridx5.i42, align 1, !tbaa !9
  %ptridx7.i43 = getelementptr inbounds i8, i8 addrspace(4)* %Buf, i64 %idxprom4.i41
  store i8 %3, i8 addrspace(4)* %ptridx7.i43, align 1, !tbaa !9
  %inc9.i44 = add nuw i32 %I.021.i40, 1
  %cmp2.i45 = icmp ult i32 %inc9.i44, %Len.0.i32
  br i1 %cmp2.i45, label %for.body3.i46, label %return, !llvm.loop !47

for.cond.i21:                                     ; preds = %if.then4, %for.cond.i21
  %Len.0.i16 = phi i32 [ %inc.i20, %for.cond.i21 ], [ 0, %if.then4 ]
  %idxprom.i17 = zext i32 %Len.0.i16 to i64
  %ptridx.i1854 = getelementptr inbounds [4 x i8], [4 x i8] addrspace(1)* @.str.4, i64 0, i64 %idxprom.i17
  %ptridx.i18 = addrspacecast i8 addrspace(1)* %ptridx.i1854 to i8 addrspace(4)*
  %4 = load i8, i8 addrspace(4)* %ptridx.i18, align 1, !tbaa !9
  %cmp.not.i19 = icmp eq i8 %4, 0
  %inc.i20 = add i32 %Len.0.i16, 1
  br i1 %cmp.not.i19, label %for.cond1.preheader.i23, label %for.cond.i21, !llvm.loop !46

for.cond1.preheader.i23:                          ; preds = %for.cond.i21
  %cmp220.not.i22 = icmp eq i32 %Len.0.i16, 0
  br i1 %cmp220.not.i22, label %return, label %for.body3.i30

for.body3.i30:                                    ; preds = %for.cond1.preheader.i23, %for.body3.i30
  %I.021.i24 = phi i32 [ %inc9.i28, %for.body3.i30 ], [ 0, %for.cond1.preheader.i23 ]
  %idxprom4.i25 = zext i32 %I.021.i24 to i64
  %ptridx5.i2655 = getelementptr inbounds [4 x i8], [4 x i8] addrspace(1)* @.str.4, i64 0, i64 %idxprom4.i25
  %ptridx5.i26 = addrspacecast i8 addrspace(1)* %ptridx5.i2655 to i8 addrspace(4)*
  %5 = load i8, i8 addrspace(4)* %ptridx5.i26, align 1, !tbaa !9
  %ptridx7.i27 = getelementptr inbounds i8, i8 addrspace(4)* %Buf, i64 %idxprom4.i25
  store i8 %5, i8 addrspace(4)* %ptridx7.i27, align 1, !tbaa !9
  %inc9.i28 = add nuw i32 %I.021.i24, 1
  %cmp2.i29 = icmp ult i32 %inc9.i28, %Len.0.i16
  br i1 %cmp2.i29, label %for.body3.i30, label %return, !llvm.loop !47

return:                                           ; preds = %for.body3.i30, %for.body3.i46, %for.body3.i, %for.cond1.preheader.i23, %for.cond1.preheader.i39, %for.cond1.preheader.i, %if.end
  %retval.0 = phi i32 [ 0, %if.end ], [ 0, %for.cond1.preheader.i ], [ 0, %for.cond1.preheader.i39 ], [ 0, %for.cond1.preheader.i23 ], [ %Len.0.i, %for.body3.i ], [ %Len.0.i32, %for.body3.i46 ], [ %Len.0.i16, %for.body3.i30 ]
  ret i32 %retval.0
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func i32 @_ZN2cl4sycl6detail21floatingPointToDecStrIfEENSt9enable_ifIXoooosr3std7is_sameIT_fEE5valuesr3std7is_sameIS4_dEE5valuesr3std7is_sameIS4_NS1_9half_impl4halfEEE5valueEjE4typeES4_Pcib(float %AbsVal, i8 addrspace(4)* %Digits, i32 %Precision, i1 zeroext %IsSci) local_unnamed_addr #6 comdat {
entry:
  %FractionDigits = alloca [24 x i32], align 4
  %cmp359 = fcmp ult float %AbsVal, 1.000000e+01
  br i1 %cmp359, label %while.cond3.preheader, label %while.body

while.cond3.preheader:                            ; preds = %while.body, %entry
  %Exp.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %AbsVal.addr.0.lcssa = phi float [ %AbsVal, %entry ], [ %conv2, %while.body ]
  %cmp5353 = fcmp ogt float %AbsVal.addr.0.lcssa, 0.000000e+00
  %cmp7354 = fcmp olt float %AbsVal.addr.0.lcssa, 1.000000e+00
  %0 = and i1 %cmp5353, %cmp7354
  br i1 %0, label %while.body8, label %while.end11

while.body:                                       ; preds = %entry, %while.body
  %AbsVal.addr.0361 = phi float [ %conv2, %while.body ], [ %AbsVal, %entry ]
  %Exp.0360 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %inc = add nuw nsw i32 %Exp.0360, 1
  %conv2 = fdiv float %AbsVal.addr.0361, 1.000000e+01
  %cmp = fcmp ult float %conv2, 1.000000e+01
  br i1 %cmp, label %while.cond3.preheader, label %while.body, !llvm.loop !48

while.body8:                                      ; preds = %while.cond3.preheader, %while.body8
  %AbsVal.addr.1356 = phi float [ %conv10, %while.body8 ], [ %AbsVal.addr.0.lcssa, %while.cond3.preheader ]
  %Exp.1355 = phi i32 [ %dec, %while.body8 ], [ %Exp.0.lcssa, %while.cond3.preheader ]
  %dec = add nsw i32 %Exp.1355, -1
  %conv10 = fmul float %AbsVal.addr.1356, 1.000000e+01
  %cmp5 = fcmp ogt float %conv10, 0.000000e+00
  %cmp7 = fcmp olt float %conv10, 1.000000e+00
  %1 = and i1 %cmp5, %cmp7
  br i1 %1, label %while.body8, label %while.end11, !llvm.loop !49

while.end11:                                      ; preds = %while.body8, %while.cond3.preheader
  %Exp.1.lcssa = phi i32 [ %Exp.0.lcssa, %while.cond3.preheader ], [ %dec, %while.body8 ]
  %AbsVal.addr.1.lcssa = phi float [ %AbsVal.addr.0.lcssa, %while.cond3.preheader ], [ %conv10, %while.body8 ]
  %conv12 = fptosi float %AbsVal.addr.1.lcssa to i32
  %conv13 = sitofp i32 %conv12 to float
  %sub = fsub float %AbsVal.addr.1.lcssa, %conv13
  %2 = bitcast [24 x i32]* %FractionDigits to i8*
  call void @llvm.lifetime.start.p0i8(i64 96, i8* nonnull %2) #8
  call void @llvm.memset.p0i8.i64(i8* nonnull align 4 dereferenceable(96) %2, i8 0, i64 96, i1 false)
  %cmp14 = icmp sgt i32 %Precision, 0
  %cond = select i1 %cmp14, i32 %Precision, i32 4
  %add = add nsw i32 %Exp.1.lcssa, %cond
  %conv15 = sext i32 %add to i64
  %3 = icmp ult i64 %conv15, 19
  %spec.store.select = select i1 %3, i64 %conv15, i64 19
  %cmp18348.not = icmp eq i64 %spec.store.select, 0
  br i1 %cmp18348.not, label %for.cond.cleanup.thread, label %for.body

for.cond.cleanup.thread:                          ; preds = %while.end11
  %cmp27369 = fcmp ogt float %sub, 5.000000e-01
  %cond28370 = zext i1 %cmp27369 to i32
  br label %for.cond.cleanup36

for.cond.cleanup:                                 ; preds = %for.body
  %cmp27 = fcmp ogt float %sub25, 5.000000e-01
  %cond28 = zext i1 %cmp27 to i32
  %4 = trunc i64 %spec.store.select to i32
  %cmp33344 = icmp ne i32 %4, 0
  %5 = and i1 %cmp27, %cmp33344
  br i1 %5, label %for.body37, label %for.cond.cleanup36

for.body:                                         ; preds = %while.end11, %for.body
  %conv17351 = phi i64 [ %conv17, %for.body ], [ 0, %while.end11 ]
  %I.0350 = phi i32 [ %inc26, %for.body ], [ 0, %while.end11 ]
  %FractionPart.0349 = phi float [ %sub25, %for.body ], [ %sub, %while.end11 ]
  %conv21 = fmul float %FractionPart.0349, 1.000000e+01
  %conv22 = fptosi float %conv21 to i32
  %arrayidx = getelementptr inbounds [24 x i32], [24 x i32]* %FractionDigits, i64 0, i64 %conv17351
  store i32 %conv22, i32* %arrayidx, align 4, !tbaa !50
  %conv24 = sitofp i32 %conv22 to float
  %sub25 = fsub float %conv21, %conv24
  %inc26 = add i32 %I.0350, 1
  %conv17 = zext i32 %inc26 to i64
  %cmp18 = icmp ugt i64 %spec.store.select, %conv17
  br i1 %cmp18, label %for.body, label %for.cond.cleanup, !llvm.loop !51

for.cond.cleanup36:                               ; preds = %for.body37, %for.cond.cleanup.thread, %for.cond.cleanup
  %Carry.0.lcssa = phi i32 [ %cond28, %for.cond.cleanup ], [ %cond28370, %for.cond.cleanup.thread ], [ %div43, %for.body37 ]
  %add47 = add nsw i32 %Carry.0.lcssa, %conv12
  %cmp48 = icmp eq i32 %add47, 10
  %spec.select = select i1 %cmp48, i32 1, i32 %add47
  %inc50 = zext i1 %cmp48 to i32
  %spec.select285 = add nsw i32 %Exp.1.lcssa, %inc50
  br i1 %IsSci, label %if.then53, label %if.else

for.body37:                                       ; preds = %for.cond.cleanup, %for.body37
  %I29.0346.in = phi i32 [ %I29.0346, %for.body37 ], [ %4, %for.cond.cleanup ]
  %Carry.0345 = phi i32 [ %div43, %for.body37 ], [ %cond28, %for.cond.cleanup ]
  %I29.0346 = add nsw i32 %I29.0346.in, -1
  %idxprom38 = sext i32 %I29.0346 to i64
  %arrayidx39 = getelementptr inbounds [24 x i32], [24 x i32]* %FractionDigits, i64 0, i64 %idxprom38
  %6 = load i32, i32* %arrayidx39, align 4, !tbaa !50
  %add40 = add nsw i32 %6, %Carry.0345
  %add40.frozen = freeze i32 %add40
  %div43 = sdiv i32 %add40.frozen, 10
  %7 = mul i32 %div43, 10
  %rem.decomposed = sub i32 %add40.frozen, %7
  store i32 %rem.decomposed, i32* %arrayidx39, align 4, !tbaa !50
  %cmp33 = icmp sgt i32 %I29.0346.in, 1
  %add40.off = add i32 %add40, 9
  %8 = icmp ugt i32 %add40.off, 18
  %9 = and i1 %8, %cmp33
  br i1 %9, label %for.body37, label %for.cond.cleanup36, !llvm.loop !52

if.then53:                                        ; preds = %for.cond.cleanup36
  %cmp.i = icmp slt i32 %spec.select, 10
  %10 = trunc i32 %spec.select to i8
  %retval.0.v.i = select i1 %cmp.i, i8 48, i8 87
  %retval.0.i = add i8 %retval.0.v.i, %10
  store i8 %retval.0.i, i8 addrspace(4)* %Digits, align 1, !tbaa !9
  %ptridx58 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 1
  store i8 46, i8 addrspace(4)* %ptridx58, align 1, !tbaa !9
  br i1 %cmp18348.not, label %for.cond.cleanup63, label %for.body64

for.cond.cleanup63:                               ; preds = %for.body64, %if.then53
  %Offset.0.lcssa = phi i32 [ 2, %if.then53 ], [ %inc68, %for.body64 ]
  %inc74 = add i32 %Offset.0.lcssa, 1
  %idxprom75 = zext i32 %Offset.0.lcssa to i64
  %ptridx76 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom75
  store i8 101, i8 addrspace(4)* %ptridx76, align 1, !tbaa !9
  %cmp77 = icmp sgt i32 %spec.select285, -1
  %cond78 = select i1 %cmp77, i8 43, i8 45
  %inc79 = add i32 %Offset.0.lcssa, 2
  %idxprom80 = zext i32 %inc74 to i64
  %ptridx81 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom80
  store i8 %cond78, i8 addrspace(4)* %ptridx81, align 1, !tbaa !9
  %call1.i.i305 = tail call i32 @_Z17__spirv_ocl_s_absi(i32 %spec.select285) #10
  %call1.i.i305.frozen = freeze i32 %call1.i.i305
  %div83 = udiv i32 %call1.i.i305.frozen, 10
  %cmp.i306 = icmp ult i32 %call1.i.i305, 100
  %11 = trunc i32 %div83 to i8
  %retval.0.v.i307 = select i1 %cmp.i306, i8 48, i8 87
  %retval.0.i308 = add i8 %retval.0.v.i307, %11
  %inc85 = add i32 %Offset.0.lcssa, 3
  %idxprom86 = zext i32 %inc79 to i64
  %ptridx87 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom86
  store i8 %retval.0.i308, i8 addrspace(4)* %ptridx87, align 1, !tbaa !9
  %12 = mul i32 %div83, 10
  %rem89.decomposed = sub i32 %call1.i.i305.frozen, %12
  %13 = trunc i32 %rem89.decomposed to i8
  %retval.0.i304 = or i8 %13, 48
  %inc91 = add i32 %Offset.0.lcssa, 4
  %idxprom92 = zext i32 %inc85 to i64
  %ptridx93 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom92
  store i8 %retval.0.i304, i8 addrspace(4)* %ptridx93, align 1, !tbaa !9
  br label %if.end186

for.body64:                                       ; preds = %if.then53, %for.body64
  %conv61315 = phi i64 [ %conv61, %for.body64 ], [ 0, %if.then53 ]
  %I59.0314 = phi i32 [ %inc72, %for.body64 ], [ 0, %if.then53 ]
  %Offset.0313 = phi i32 [ %inc68, %for.body64 ], [ 2, %if.then53 ]
  %arrayidx66 = getelementptr inbounds [24 x i32], [24 x i32]* %FractionDigits, i64 0, i64 %conv61315
  %14 = load i32, i32* %arrayidx66, align 4, !tbaa !50
  %cmp.i301 = icmp slt i32 %14, 10
  %15 = trunc i32 %14 to i8
  %retval.0.v.i302 = select i1 %cmp.i301, i8 48, i8 87
  %retval.0.i303 = add i8 %retval.0.v.i302, %15
  %inc68 = add i32 %Offset.0313, 1
  %idxprom69 = zext i32 %Offset.0313 to i64
  %ptridx70 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom69
  store i8 %retval.0.i303, i8 addrspace(4)* %ptridx70, align 1, !tbaa !9
  %inc72 = add i32 %I59.0314, 1
  %conv61 = zext i32 %inc72 to i64
  %cmp62 = icmp ugt i64 %spec.store.select, %conv61
  br i1 %cmp62, label %for.body64, label %for.cond.cleanup63, !llvm.loop !53

if.else:                                          ; preds = %for.cond.cleanup36
  %cmp94 = icmp slt i32 %spec.select285, 0
  br i1 %cmp94, label %if.then95, label %if.else129

if.then95:                                        ; preds = %if.else
  store i8 48, i8 addrspace(4)* %Digits, align 1, !tbaa !9
  %ptridx101 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 1
  store i8 46, i8 addrspace(4)* %ptridx101, align 1, !tbaa !9
  %inc103322 = add nsw i32 %spec.select285, 1
  %tobool104.not323 = icmp eq i32 %inc103322, 0
  br i1 %tobool104.not323, label %while.end109, label %while.body105

while.body105:                                    ; preds = %if.then95, %while.body105
  %inc103325 = phi i32 [ %inc103, %while.body105 ], [ %inc103322, %if.then95 ]
  %Offset.1324 = phi i32 [ %inc106, %while.body105 ], [ 2, %if.then95 ]
  %inc106 = add i32 %Offset.1324, 1
  %idxprom107 = zext i32 %Offset.1324 to i64
  %ptridx108 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom107
  store i8 48, i8 addrspace(4)* %ptridx108, align 1, !tbaa !9
  %inc103 = add nsw i32 %inc103325, 1
  %tobool104.not = icmp eq i32 %inc103, 0
  br i1 %tobool104.not, label %while.end109, label %while.body105, !llvm.loop !54

while.end109:                                     ; preds = %while.body105, %if.then95
  %Offset.1.lcssa = phi i32 [ 2, %if.then95 ], [ %inc106, %while.body105 ]
  %cmp.i298 = icmp slt i32 %spec.select, 10
  %16 = trunc i32 %spec.select to i8
  %retval.0.v.i299 = select i1 %cmp.i298, i8 48, i8 87
  %retval.0.i300 = add i8 %retval.0.v.i299, %16
  %idxprom112 = zext i32 %Offset.1.lcssa to i64
  %ptridx113 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom112
  store i8 %retval.0.i300, i8 addrspace(4)* %ptridx113, align 1, !tbaa !9
  %Offset.2316 = add i32 %Offset.1.lcssa, 1
  br i1 %cmp18348.not, label %while.cond169.preheader, label %for.body119

for.body119:                                      ; preds = %while.end109, %for.body119
  %conv116320 = phi i64 [ %conv116, %for.body119 ], [ 0, %while.end109 ]
  %Offset.2319 = phi i32 [ %Offset.2, %for.body119 ], [ %Offset.2316, %while.end109 ]
  %I114.0318 = phi i32 [ %inc127, %for.body119 ], [ 0, %while.end109 ]
  %arrayidx121 = getelementptr inbounds [24 x i32], [24 x i32]* %FractionDigits, i64 0, i64 %conv116320
  %17 = load i32, i32* %arrayidx121, align 4, !tbaa !50
  %cmp.i295 = icmp slt i32 %17, 10
  %18 = trunc i32 %17 to i8
  %retval.0.v.i296 = select i1 %cmp.i295, i8 48, i8 87
  %retval.0.i297 = add i8 %retval.0.v.i296, %18
  %idxprom124 = zext i32 %Offset.2319 to i64
  %ptridx125 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom124
  store i8 %retval.0.i297, i8 addrspace(4)* %ptridx125, align 1, !tbaa !9
  %inc127 = add i32 %I114.0318, 1
  %Offset.2 = add i32 %Offset.2319, 1
  %conv116 = zext i32 %inc127 to i64
  %cmp117 = icmp ugt i64 %spec.store.select, %conv116
  br i1 %cmp117, label %for.body119, label %while.cond169.preheader, !llvm.loop !55

if.else129:                                       ; preds = %if.else
  %cmp.i292 = icmp slt i32 %spec.select, 10
  %19 = trunc i32 %spec.select to i8
  %retval.0.v.i293 = select i1 %cmp.i292, i8 48, i8 87
  %retval.0.i294 = add i8 %retval.0.v.i293, %19
  store i8 %retval.0.i294, i8 addrspace(4)* %Digits, align 1, !tbaa !9
  %tobool140.not335 = icmp eq i32 %spec.select285, 0
  %or.cond336 = or i1 %tobool140.not335, %cmp18348.not
  br i1 %or.cond336, label %for.end151, label %for.body142

for.body142:                                      ; preds = %if.else129, %for.body142
  %conv136340 = phi i64 [ %conv136, %for.body142 ], [ 0, %if.else129 ]
  %I134.0339 = phi i32 [ %inc150, %for.body142 ], [ 0, %if.else129 ]
  %Offset.3338 = phi i32 [ %inc146, %for.body142 ], [ 1, %if.else129 ]
  %Exp.4337 = phi i32 [ %dec139, %for.body142 ], [ %spec.select285, %if.else129 ]
  %dec139 = add nsw i32 %Exp.4337, -1
  %arrayidx144 = getelementptr inbounds [24 x i32], [24 x i32]* %FractionDigits, i64 0, i64 %conv136340
  %20 = load i32, i32* %arrayidx144, align 4, !tbaa !50
  %cmp.i289 = icmp slt i32 %20, 10
  %21 = trunc i32 %20 to i8
  %retval.0.v.i290 = select i1 %cmp.i289, i8 48, i8 87
  %retval.0.i291 = add i8 %retval.0.v.i290, %21
  %inc146 = add i32 %Offset.3338, 1
  %idxprom147 = zext i32 %Offset.3338 to i64
  %ptridx148 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom147
  store i8 %retval.0.i291, i8 addrspace(4)* %ptridx148, align 1, !tbaa !9
  %inc150 = add i32 %I134.0339, 1
  %conv136 = zext i32 %inc150 to i64
  %cmp137 = icmp ule i64 %spec.store.select, %conv136
  %tobool140.not = icmp eq i32 %dec139, 0
  %or.cond = or i1 %tobool140.not, %cmp137
  br i1 %or.cond, label %for.end151, label %for.body142, !llvm.loop !56

for.end151:                                       ; preds = %for.body142, %if.else129
  %Offset.3.lcssa = phi i32 [ 1, %if.else129 ], [ %inc146, %for.body142 ]
  %I134.0.lcssa = phi i32 [ 0, %if.else129 ], [ %inc150, %for.body142 ]
  %idxprom153 = zext i32 %Offset.3.lcssa to i64
  %ptridx154 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom153
  store i8 46, i8 addrspace(4)* %ptridx154, align 1, !tbaa !9
  %Offset.4327 = add i32 %Offset.3.lcssa, 1
  %conv156328 = zext i32 %I134.0.lcssa to i64
  %cmp157329 = icmp ugt i64 %spec.store.select, %conv156328
  br i1 %cmp157329, label %for.body158, label %while.cond169.preheader

for.body158:                                      ; preds = %for.end151, %for.body158
  %conv156332 = phi i64 [ %conv156, %for.body158 ], [ %conv156328, %for.end151 ]
  %Offset.4331 = phi i32 [ %Offset.4, %for.body158 ], [ %Offset.4327, %for.end151 ]
  %I134.1330 = phi i32 [ %inc166, %for.body158 ], [ %I134.0.lcssa, %for.end151 ]
  %arrayidx160 = getelementptr inbounds [24 x i32], [24 x i32]* %FractionDigits, i64 0, i64 %conv156332
  %22 = load i32, i32* %arrayidx160, align 4, !tbaa !50
  %cmp.i286 = icmp slt i32 %22, 10
  %23 = trunc i32 %22 to i8
  %retval.0.v.i287 = select i1 %cmp.i286, i8 48, i8 87
  %retval.0.i288 = add i8 %retval.0.v.i287, %23
  %idxprom163 = zext i32 %Offset.4331 to i64
  %ptridx164 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom163
  store i8 %retval.0.i288, i8 addrspace(4)* %ptridx164, align 1, !tbaa !9
  %inc166 = add i32 %I134.1330, 1
  %Offset.4 = add i32 %Offset.4331, 1
  %conv156 = zext i32 %inc166 to i64
  %cmp157 = icmp ugt i64 %spec.store.select, %conv156
  br i1 %cmp157, label %for.body158, label %while.cond169.preheader, !llvm.loop !57

while.cond169.preheader:                          ; preds = %for.body158, %for.body119, %for.end151, %while.end109
  %Offset.6.ph = phi i32 [ %Offset.4327, %for.end151 ], [ %Offset.2316, %while.end109 ], [ %Offset.2, %for.body119 ], [ %Offset.4, %for.body158 ]
  br label %while.cond169

while.cond169:                                    ; preds = %while.cond169.preheader, %while.cond169
  %Offset.6 = phi i32 [ %sub170, %while.cond169 ], [ %Offset.6.ph, %while.cond169.preheader ]
  %sub170 = add i32 %Offset.6, -1
  %idxprom171 = zext i32 %sub170 to i64
  %ptridx172 = getelementptr inbounds i8, i8 addrspace(4)* %Digits, i64 %idxprom171
  %24 = load i8, i8 addrspace(4)* %ptridx172, align 1, !tbaa !9
  switch i8 %24, label %if.end186.loopexit [
    i8 48, label %while.cond169
    i8 46, label %if.end186
  ], !llvm.loop !58

if.end186.loopexit:                               ; preds = %while.cond169
  br label %if.end186

if.end186:                                        ; preds = %while.cond169, %if.end186.loopexit, %for.cond.cleanup63
  %Offset.7 = phi i32 [ %inc91, %for.cond.cleanup63 ], [ %Offset.6, %if.end186.loopexit ], [ %sub170, %while.cond169 ]
  call void @llvm.lifetime.end.p0i8(i64 96, i8* nonnull %2) #8
  ret i32 %Offset.7
}

; Function Attrs: convergent nounwind readnone
declare dso_local i32 @_Z17__spirv_ocl_s_absi(i32) local_unnamed_addr #7

; Function Attrs: convergent nounwind readnone
declare dso_local zeroext i1 @_Z13__spirv_IsNanf(float) local_unnamed_addr #7

; Function Attrs: convergent nounwind readnone
declare dso_local zeroext i1 @_Z13__spirv_IsInff(float) local_unnamed_addr #7

; Function Attrs: convergent nounwind readnone
declare dso_local zeroext i1 @_Z18__spirv_SignBitSetf(float) local_unnamed_addr #7

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSN4test8kernel_tIiEE(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* byval(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream") align 8 %_arg_strm_, i8 addrspace(1)* %_arg_GlobalBuf, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalBuf1, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalBuf2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalBuf3, i32 addrspace(1)* %_arg_GlobalOffset, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalOffset4, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalOffset5, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalOffset6, i8 addrspace(1)* %_arg_GlobalFlushBuf, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalFlushBuf7, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalFlushBuf8, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_GlobalFlushBuf9) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %kernel_t = alloca %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", align 8
  %agg.tmp22 = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", align 8
  %0 = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 0, i32 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 152, i8* nonnull %0) #8
  %strm_ = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 1
  %1 = getelementptr %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* %strm_, i64 0, i32 0, i64 0
  %2 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* %_arg_strm_, i64 0, i32 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(144) %1, i8* nonnull align 8 dereferenceable(144) %2, i64 144, i1 false)
  %GlobalBuf = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 1, i32 1
  %3 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalBuf1, i64 0, i32 0, i32 0, i64 0
  %4 = load i64, i64* %3, align 8
  %5 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalBuf2, i64 0, i32 0, i32 0, i64 0
  %6 = load i64, i64* %5, align 8
  %7 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalBuf3, i64 0, i32 0, i32 0, i64 0
  %8 = load i64, i64* %7, align 8
  %9 = addrspacecast %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* %GlobalBuf to %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*
  %MData.i = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %9, i64 0, i32 1, i32 0
  %arrayidx.i25.i = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %9, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
  store i64 %8, i64 addrspace(4)* %arrayidx.i25.i, align 8, !tbaa !5
  %arrayidx.i23.i = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %9, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  store i64 %4, i64 addrspace(4)* %arrayidx.i23.i, align 8, !tbaa !5
  %arrayidx.i21.i = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %9, i64 0, i32 0, i32 2, i32 0, i32 0, i64 0
  store i64 %6, i64 addrspace(4)* %arrayidx.i21.i, align 8, !tbaa !5
  %add.ptr.i = getelementptr inbounds i8, i8 addrspace(1)* %_arg_GlobalBuf, i64 %8
  store i8 addrspace(1)* %add.ptr.i, i8 addrspace(1)* addrspace(4)* %MData.i, align 8, !tbaa !9
  %GlobalOffset = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 1, i32 2
  %10 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalOffset4, i64 0, i32 0, i32 0, i64 0
  %11 = load i64, i64* %10, align 8
  %12 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalOffset5, i64 0, i32 0, i32 0, i64 0
  %13 = load i64, i64* %12, align 8
  %14 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalOffset6, i64 0, i32 0, i32 0, i64 0
  %15 = load i64, i64* %14, align 8
  %16 = addrspacecast %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* %GlobalOffset to %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*
  %MData.i30 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %16, i64 0, i32 1, i32 0
  %arrayidx.i25.i32 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %16, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
  store i64 %15, i64 addrspace(4)* %arrayidx.i25.i32, align 8, !tbaa !5
  %arrayidx.i23.i34 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %16, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  store i64 %11, i64 addrspace(4)* %arrayidx.i23.i34, align 8, !tbaa !5
  %arrayidx.i21.i36 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %16, i64 0, i32 0, i32 2, i32 0, i32 0, i64 0
  store i64 %13, i64 addrspace(4)* %arrayidx.i21.i36, align 8, !tbaa !5
  %add.ptr.i37 = getelementptr inbounds i32, i32 addrspace(1)* %_arg_GlobalOffset, i64 %15
  store i32 addrspace(1)* %add.ptr.i37, i32 addrspace(1)* addrspace(4)* %MData.i30, align 8, !tbaa !9
  %GlobalFlushBuf = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t, i64 0, i32 1, i32 3
  %17 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalFlushBuf7, i64 0, i32 0, i32 0, i64 0
  %18 = load i64, i64* %17, align 8
  %19 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalFlushBuf8, i64 0, i32 0, i32 0, i64 0
  %20 = load i64, i64* %19, align 8
  %21 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_GlobalFlushBuf9, i64 0, i32 0, i32 0, i64 0
  %22 = load i64, i64* %21, align 8
  %23 = addrspacecast %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor"* %GlobalFlushBuf to %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*
  %MData.i41 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %23, i64 0, i32 1, i32 0
  %arrayidx.i25.i43 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %23, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
  store i64 %22, i64 addrspace(4)* %arrayidx.i25.i43, align 8, !tbaa !5
  %arrayidx.i23.i45 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %23, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  store i64 %18, i64 addrspace(4)* %arrayidx.i23.i45, align 8, !tbaa !5
  %arrayidx.i21.i47 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %23, i64 0, i32 0, i32 2, i32 0, i32 0, i64 0
  store i64 %20, i64 addrspace(4)* %arrayidx.i21.i47, align 8, !tbaa !5
  %add.ptr.i48 = getelementptr inbounds i8, i8 addrspace(1)* %_arg_GlobalFlushBuf, i64 %22
  store i8 addrspace(1)* %add.ptr.i48, i8 addrspace(1)* addrspace(4)* %MData.i41, align 8, !tbaa !9
  %24 = addrspacecast %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream"* %strm_ to %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)*
  %add.ptr.i.i = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr.i37, i64 1
  %FlushBufferSize.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %24, i64 0, i32 6
  %25 = load i64, i64 addrspace(4)* %FlushBufferSize.i, align 8, !tbaa !10
  %conv.i = trunc i64 %25 to i32
  %call2.i.i = tail call spir_func i32 @_Z18__spirv_AtomicIAddPU3AS1jN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEj(i32 addrspace(1)* %add.ptr.i.i, i32 1, i32 0, i32 %conv.i) #9
  %WIOffset.i = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %24, i64 0, i32 4
  store i32 %call2.i.i, i32 addrspace(4)* %WIOffset.i, align 8, !tbaa !19
  %conv1.i.i = zext i32 %call2.i.i to i64
  %ptridx.i14.i.i = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr.i48, i64 %conv1.i.i
  %ptridx.ascast.i15.i.i = addrspacecast i8 addrspace(1)* %ptridx.i14.i.i to i8 addrspace(4)*
  store i8 0, i8 addrspace(4)* %ptridx.ascast.i15.i.i, align 1, !tbaa !9
  %add.i.i = add i32 %call2.i.i, 1
  %conv5.i.i = zext i32 %add.i.i to i64
  %ptridx.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr.i48, i64 %conv5.i.i
  %ptridx.ascast.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i.i to i8 addrspace(4)*
  store i8 0, i8 addrspace(4)* %ptridx.ascast.i.i.i, align 1, !tbaa !9
  %26 = addrspacecast %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %agg.tmp22 to %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)*
  %27 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !59
  %28 = extractelement <3 x i64> %27, i64 0
  %arrayinit.begin.i.i.i.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" addrspace(4)* %26, i64 0, i32 0, i32 0, i64 0
  store i64 %28, i64 addrspace(4)* %arrayinit.begin.i.i.i.i.i, align 8, !tbaa !5, !alias.scope !59
  %29 = addrspacecast %"class._ZTSN4test8kernel_tIfEE.test::kernel_t"* %kernel_t to %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)*
  call spir_func void @_ZNK4test8kernel_tIiEclEN2cl4sycl2idILi1EEE(%"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* dereferenceable_or_null(152) %29, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* nonnull byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %agg.tmp22) #9
  call spir_func void @_ZN2cl4sycl6stream10__finalizeEv(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* dereferenceable_or_null(144) %24) #9
  call void @llvm.lifetime.end.p0i8(i64 152, i8* nonnull %0) #8
  ret void
}

; Function Attrs: convergent norecurse
define linkonce_odr dso_local spir_func void @_ZNK4test8kernel_tIiEclEN2cl4sycl2idILi1EEE(%"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* dereferenceable_or_null(152) %this, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %i) local_unnamed_addr #3 comdat align 2 {
entry:
  %ref.tmp = alloca %"struct._ZTSN4test5pod_tE.test::pod_t", align 4
  %strm_ = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.cond.i, %entry
  %Len.0.i = phi i32 [ 0, %entry ], [ %inc.i, %for.cond.i ]
  %idxprom.i = zext i32 %Len.0.i to i64
  %ptridx.i62 = getelementptr inbounds [11 x i8], [11 x i8] addrspace(1)* @.str, i64 0, i64 %idxprom.i
  %ptridx.i = addrspacecast i8 addrspace(1)* %ptridx.i62 to i8 addrspace(4)*
  %0 = load i8, i8 addrspace(4)* %ptridx.i, align 1, !tbaa !9
  %cmp.not.i = icmp eq i8 %0, 0
  %inc.i = add i32 %Len.0.i, 1
  br i1 %cmp.not.i, label %for.end.i, label %for.cond.i, !llvm.loop !27

for.end.i:                                        ; preds = %for.cond.i
  %FlushBufferSize.i = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1, i32 6
  %1 = load i64, i64 addrspace(4)* %FlushBufferSize.i, align 8, !tbaa !10
  %WIOffset.i = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1, i32 4
  %2 = load i32, i32 addrspace(4)* %WIOffset.i, align 8, !tbaa !19
  %conv.i.i.i = zext i32 %2 to i64
  %MData.i.i12.i.i.i = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1, i32 3, i32 1, i32 0
  %3 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i, align 8, !tbaa !9
  %ptridx.i13.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %conv.i.i.i
  %ptridx.ascast.i14.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i13.i.i.i to i8 addrspace(4)*
  %4 = load i8, i8 addrspace(4)* %ptridx.ascast.i14.i.i.i, align 1, !tbaa !9
  %conv1.i.i.i = zext i8 %4 to i32
  %shl.i.i.i = shl nuw nsw i32 %conv1.i.i.i, 8
  %add.i.i.i = add i32 %2, 1
  %conv3.i.i.i = zext i32 %add.i.i.i to i64
  %ptridx.i.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %conv3.i.i.i
  %ptridx.ascast.i.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i.i.i to i8 addrspace(4)*
  %5 = load i8, i8 addrspace(4)* %ptridx.ascast.i.i.i.i, align 1, !tbaa !9
  %conv5.i.i.i = zext i8 %5 to i32
  %add6.i.i.i = or i32 %shl.i.i.i, %conv5.i.i.i
  %add.i.i = add nuw nsw i32 %add6.i.i.i, 2
  %add2.i.i = add i32 %add.i.i, %Len.0.i
  %conv.i.i = zext i32 %add2.i.i to i64
  %cmp.i.i = icmp ult i64 %1, %conv.i.i
  br i1 %cmp.i.i, label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit, label %lor.lhs.false.i.i

lor.lhs.false.i.i:                                ; preds = %for.end.i
  %add5.i.i = add i32 %add2.i.i, %2
  %conv6.i.i = zext i32 %add5.i.i to i64
  %arrayidx.i.i.i.i.i = getelementptr inbounds %"class._ZTSN4test8kernel_tIfEE.test::kernel_t", %"class._ZTSN4test8kernel_tIfEE.test::kernel_t" addrspace(4)* %this, i64 0, i32 1, i32 3, i32 0, i32 1, i32 0, i32 0, i64 0
  %6 = load i64, i64 addrspace(4)* %arrayidx.i.i.i.i.i, align 8, !tbaa !5
  %cmp8.i.i = icmp ult i64 %6, %conv6.i.i
  br i1 %cmp8.i.i, label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit, label %for.cond.preheader.i.i

for.cond.preheader.i.i:                           ; preds = %lor.lhs.false.i.i
  %cmp1868.not.i.i = icmp eq i32 %Len.0.i, 0
  br i1 %cmp1868.not.i.i, label %for.cond.cleanup19.i.i, label %for.body20.i.i.preheader

for.body20.i.i.preheader:                         ; preds = %for.cond.preheader.i.i
  %7 = load i8, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(1)* @.str, i64 0, i64 0) to i8 addrspace(4)*), align 1, !tbaa !9
  %add22.i.i74 = add i32 %add.i.i, %2
  %conv23.i.i75 = zext i32 %add22.i.i74 to i64
  %ptridx.i.i.i76 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %conv23.i.i75
  %ptridx.ascast.i.i.i77 = addrspacecast i8 addrspace(1)* %ptridx.i.i.i76 to i8 addrspace(4)*
  store i8 %7, i8 addrspace(4)* %ptridx.ascast.i.i.i77, align 1, !tbaa !9
  %inc27.i.i78 = add nuw nsw i32 %add6.i.i.i, 3
  %cmp18.i.i79.not = icmp eq i32 %Len.0.i, 1
  br i1 %cmp18.i.i79.not, label %for.cond.cleanup19.i.loopexit.i, label %for.body20.i.for.body20.i_crit_edge.i, !llvm.loop !29

for.cond.cleanup19.i.loopexit.i:                  ; preds = %for.body20.i.for.body20.i_crit_edge.i, %for.body20.i.i.preheader
  %inc27.i.i.lcssa = phi i32 [ %inc27.i.i78, %for.body20.i.i.preheader ], [ %inc27.i.i, %for.body20.i.for.body20.i_crit_edge.i ]
  %.pre8.i = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i, align 8, !tbaa !9
  br label %for.cond.cleanup19.i.i

for.cond.cleanup19.i.i:                           ; preds = %for.cond.cleanup19.i.loopexit.i, %for.cond.preheader.i.i
  %8 = phi i8 addrspace(1)* [ %3, %for.cond.preheader.i.i ], [ %.pre8.i, %for.cond.cleanup19.i.loopexit.i ]
  %Offset.1.lcssa.i.i = phi i32 [ %add.i.i, %for.cond.preheader.i.i ], [ %inc27.i.i.lcssa, %for.cond.cleanup19.i.loopexit.i ]
  %sub.i.i = add i32 %Offset.1.lcssa.i.i, -2
  %shr.i.i.i = lshr i32 %sub.i.i, 8
  %conv.i54.i.i = trunc i32 %shr.i.i.i to i8
  %ptridx.i14.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %8, i64 %conv.i.i.i
  %ptridx.ascast.i15.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i14.i.i.i to i8 addrspace(4)*
  store i8 %conv.i54.i.i, i8 addrspace(4)* %ptridx.ascast.i15.i.i.i, align 1, !tbaa !9
  %conv3.i56.i.i = trunc i32 %sub.i.i to i8
  %9 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i, align 8, !tbaa !9
  %ptridx.i.i59.i.i = getelementptr inbounds i8, i8 addrspace(1)* %9, i64 %conv3.i.i.i
  %ptridx.ascast.i.i60.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i59.i.i to i8 addrspace(4)*
  store i8 %conv3.i56.i.i, i8 addrspace(4)* %ptridx.ascast.i.i60.i.i, align 1, !tbaa !9
  br label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit

for.body20.i.for.body20.i_crit_edge.i:            ; preds = %for.body20.i.i.preheader, %for.body20.i.for.body20.i_crit_edge.i
  %inc27.i.i81 = phi i32 [ %inc27.i.i, %for.body20.i.for.body20.i_crit_edge.i ], [ %inc27.i.i78, %for.body20.i.i.preheader ]
  %inc26.i.i80 = phi i64 [ %inc26.i.i, %for.body20.i.for.body20.i_crit_edge.i ], [ 1, %for.body20.i.i.preheader ]
  %.pre.i = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i, align 8, !tbaa !9
  %ptridx.i.i63 = getelementptr inbounds [11 x i8], [11 x i8] addrspace(1)* @.str, i64 0, i64 %inc26.i.i80
  %ptridx.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i63 to i8 addrspace(4)*
  %10 = load i8, i8 addrspace(4)* %ptridx.i.i, align 1, !tbaa !9
  %add22.i.i = add i32 %inc27.i.i81, %2
  %conv23.i.i = zext i32 %add22.i.i to i64
  %ptridx.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %.pre.i, i64 %conv23.i.i
  %ptridx.ascast.i.i.i = addrspacecast i8 addrspace(1)* %ptridx.i.i.i to i8 addrspace(4)*
  store i8 %10, i8 addrspace(4)* %ptridx.ascast.i.i.i, align 1, !tbaa !9
  %inc26.i.i = add nuw i64 %inc26.i.i80, 1
  %inc27.i.i = add i32 %inc27.i.i81, 1
  %cmp18.i.i = icmp ult i64 %inc26.i.i, %idxprom.i
  br i1 %cmp18.i.i, label %for.body20.i.for.body20.i_crit_edge.i, label %for.cond.cleanup19.i.loopexit.i, !llvm.loop !29

_ZN2cl4sycllsERKNS0_6streamEPKc.exit:             ; preds = %for.end.i, %lor.lhs.false.i.i, %for.cond.cleanup19.i.i
  %11 = bitcast %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %11) #8
  %12 = addrspacecast %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp to %"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)*
  call spir_func void @_Z36__sycl_getCompositeSpecConstantValueIN4test5pod_tEET_PKc(%"struct._ZTSN4test5pod_tE.test::pod_t" addrspace(4)* sret(%"struct._ZTSN4test5pod_tE.test::pod_t") align 4 %12, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([18 x i8], [18 x i8] addrspace(1)* @__builtin_unique_stable_name._ZNK2cl4sycl6ONEAPI12experimental13spec_constantIN4test5pod_tE11sc_kernel_tE3getIS5_EENSt9enable_ifIXaasr3std8is_classIT_EE5valuesr3std6is_podISA_EE5valueESA_E4typeEv, i64 0, i64 0) to i8 addrspace(4)*)) #9
  %x = getelementptr inbounds %"struct._ZTSN4test5pod_tE.test::pod_t", %"struct._ZTSN4test5pod_tE.test::pod_t"* %ref.tmp, i64 0, i32 0
  %13 = addrspacecast float* %x to float addrspace(4)*
  %call2 = call spir_func align 8 dereferenceable(144) %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* @_ZN2cl4sycllsERKNS0_6streamERKf(%"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* align 8 dereferenceable(144) %strm_, float addrspace(4)* align 4 dereferenceable(4) %13) #9
  br label %for.cond.i9

for.cond.i9:                                      ; preds = %for.cond.i9, %_ZN2cl4sycllsERKNS0_6streamEPKc.exit
  %Len.0.i4 = phi i32 [ 0, %_ZN2cl4sycllsERKNS0_6streamEPKc.exit ], [ %inc.i8, %for.cond.i9 ]
  %idxprom.i5 = zext i32 %Len.0.i4 to i64
  %ptridx.i664 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(1)* @.str.1, i64 0, i64 %idxprom.i5
  %ptridx.i6 = addrspacecast i8 addrspace(1)* %ptridx.i664 to i8 addrspace(4)*
  %14 = load i8, i8 addrspace(4)* %ptridx.i6, align 1, !tbaa !9
  %cmp.not.i7 = icmp eq i8 %14, 0
  %inc.i8 = add i32 %Len.0.i4, 1
  br i1 %cmp.not.i7, label %for.end.i28, label %for.cond.i9, !llvm.loop !27

for.end.i28:                                      ; preds = %for.cond.i9
  %FlushBufferSize.i10 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %call2, i64 0, i32 6
  %15 = load i64, i64 addrspace(4)* %FlushBufferSize.i10, align 8, !tbaa !10
  %WIOffset.i11 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %call2, i64 0, i32 4
  %16 = load i32, i32 addrspace(4)* %WIOffset.i11, align 8, !tbaa !19
  %conv.i.i.i12 = zext i32 %16 to i64
  %MData.i.i12.i.i.i13 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %call2, i64 0, i32 3, i32 1, i32 0
  %17 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i13, align 8, !tbaa !9
  %ptridx.i13.i.i.i14 = getelementptr inbounds i8, i8 addrspace(1)* %17, i64 %conv.i.i.i12
  %ptridx.ascast.i14.i.i.i15 = addrspacecast i8 addrspace(1)* %ptridx.i13.i.i.i14 to i8 addrspace(4)*
  %18 = load i8, i8 addrspace(4)* %ptridx.ascast.i14.i.i.i15, align 1, !tbaa !9
  %conv1.i.i.i16 = zext i8 %18 to i32
  %shl.i.i.i17 = shl nuw nsw i32 %conv1.i.i.i16, 8
  %add.i.i.i18 = add i32 %16, 1
  %conv3.i.i.i19 = zext i32 %add.i.i.i18 to i64
  %ptridx.i.i.i.i20 = getelementptr inbounds i8, i8 addrspace(1)* %17, i64 %conv3.i.i.i19
  %ptridx.ascast.i.i.i.i21 = addrspacecast i8 addrspace(1)* %ptridx.i.i.i.i20 to i8 addrspace(4)*
  %19 = load i8, i8 addrspace(4)* %ptridx.ascast.i.i.i.i21, align 1, !tbaa !9
  %conv5.i.i.i22 = zext i8 %19 to i32
  %add6.i.i.i23 = or i32 %shl.i.i.i17, %conv5.i.i.i22
  %add.i.i24 = add nuw nsw i32 %add6.i.i.i23, 2
  %add2.i.i25 = add i32 %add.i.i24, %Len.0.i4
  %conv.i.i26 = zext i32 %add2.i.i25 to i64
  %cmp.i.i27 = icmp ult i64 %15, %conv.i.i26
  br i1 %cmp.i.i27, label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit61, label %lor.lhs.false.i.i33

lor.lhs.false.i.i33:                              ; preds = %for.end.i28
  %add5.i.i29 = add i32 %add2.i.i25, %16
  %conv6.i.i30 = zext i32 %add5.i.i29 to i64
  %arrayidx.i.i.i.i.i31 = getelementptr inbounds %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream", %"class._ZTSN2cl4sycl6streamE.cl::sycl::stream" addrspace(4)* %call2, i64 0, i32 3, i32 0, i32 1, i32 0, i32 0, i64 0
  %20 = load i64, i64 addrspace(4)* %arrayidx.i.i.i.i.i31, align 8, !tbaa !5
  %cmp8.i.i32 = icmp ult i64 %20, %conv6.i.i30
  br i1 %cmp8.i.i32, label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit61, label %for.cond.preheader.i.i35

for.cond.preheader.i.i35:                         ; preds = %lor.lhs.false.i.i33
  %cmp1868.not.i.i34 = icmp eq i32 %Len.0.i4, 0
  br i1 %cmp1868.not.i.i34, label %for.cond.cleanup19.i.i47, label %for.body20.i.i58.preheader

for.body20.i.i58.preheader:                       ; preds = %for.cond.preheader.i.i35
  %21 = load i8, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(1)* @.str.1, i64 0, i64 0) to i8 addrspace(4)*), align 1, !tbaa !9
  %add22.i.i5166 = add i32 %add.i.i24, %16
  %conv23.i.i5267 = zext i32 %add22.i.i5166 to i64
  %ptridx.i.i.i5368 = getelementptr inbounds i8, i8 addrspace(1)* %17, i64 %conv23.i.i5267
  %ptridx.ascast.i.i.i5469 = addrspacecast i8 addrspace(1)* %ptridx.i.i.i5368 to i8 addrspace(4)*
  store i8 %21, i8 addrspace(4)* %ptridx.ascast.i.i.i5469, align 1, !tbaa !9
  %inc27.i.i5670 = add nuw nsw i32 %add6.i.i.i23, 3
  %cmp18.i.i5771.not = icmp eq i32 %Len.0.i4, 1
  br i1 %cmp18.i.i5771.not, label %for.cond.cleanup19.i.loopexit.i37, label %for.body20.i.for.body20.i_crit_edge.i60, !llvm.loop !29

for.cond.cleanup19.i.loopexit.i37:                ; preds = %for.body20.i.for.body20.i_crit_edge.i60, %for.body20.i.i58.preheader
  %inc27.i.i56.lcssa = phi i32 [ %inc27.i.i5670, %for.body20.i.i58.preheader ], [ %inc27.i.i56, %for.body20.i.for.body20.i_crit_edge.i60 ]
  %.pre8.i36 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i13, align 8, !tbaa !9
  br label %for.cond.cleanup19.i.i47

for.cond.cleanup19.i.i47:                         ; preds = %for.cond.cleanup19.i.loopexit.i37, %for.cond.preheader.i.i35
  %22 = phi i8 addrspace(1)* [ %17, %for.cond.preheader.i.i35 ], [ %.pre8.i36, %for.cond.cleanup19.i.loopexit.i37 ]
  %Offset.1.lcssa.i.i38 = phi i32 [ %add.i.i24, %for.cond.preheader.i.i35 ], [ %inc27.i.i56.lcssa, %for.cond.cleanup19.i.loopexit.i37 ]
  %sub.i.i39 = add i32 %Offset.1.lcssa.i.i38, -2
  %shr.i.i.i40 = lshr i32 %sub.i.i39, 8
  %conv.i54.i.i41 = trunc i32 %shr.i.i.i40 to i8
  %ptridx.i14.i.i.i42 = getelementptr inbounds i8, i8 addrspace(1)* %22, i64 %conv.i.i.i12
  %ptridx.ascast.i15.i.i.i43 = addrspacecast i8 addrspace(1)* %ptridx.i14.i.i.i42 to i8 addrspace(4)*
  store i8 %conv.i54.i.i41, i8 addrspace(4)* %ptridx.ascast.i15.i.i.i43, align 1, !tbaa !9
  %conv3.i56.i.i44 = trunc i32 %sub.i.i39 to i8
  %23 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i13, align 8, !tbaa !9
  %ptridx.i.i59.i.i45 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %conv3.i.i.i19
  %ptridx.ascast.i.i60.i.i46 = addrspacecast i8 addrspace(1)* %ptridx.i.i59.i.i45 to i8 addrspace(4)*
  store i8 %conv3.i56.i.i44, i8 addrspace(4)* %ptridx.ascast.i.i60.i.i46, align 1, !tbaa !9
  br label %_ZN2cl4sycllsERKNS0_6streamEPKc.exit61

for.body20.i.for.body20.i_crit_edge.i60:          ; preds = %for.body20.i.i58.preheader, %for.body20.i.for.body20.i_crit_edge.i60
  %inc27.i.i5673 = phi i32 [ %inc27.i.i56, %for.body20.i.for.body20.i_crit_edge.i60 ], [ %inc27.i.i5670, %for.body20.i.i58.preheader ]
  %inc26.i.i5572 = phi i64 [ %inc26.i.i55, %for.body20.i.for.body20.i_crit_edge.i60 ], [ 1, %for.body20.i.i58.preheader ]
  %.pre.i59 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %MData.i.i12.i.i.i13, align 8, !tbaa !9
  %ptridx.i.i5065 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(1)* @.str.1, i64 0, i64 %inc26.i.i5572
  %ptridx.i.i50 = addrspacecast i8 addrspace(1)* %ptridx.i.i5065 to i8 addrspace(4)*
  %24 = load i8, i8 addrspace(4)* %ptridx.i.i50, align 1, !tbaa !9
  %add22.i.i51 = add i32 %inc27.i.i5673, %16
  %conv23.i.i52 = zext i32 %add22.i.i51 to i64
  %ptridx.i.i.i53 = getelementptr inbounds i8, i8 addrspace(1)* %.pre.i59, i64 %conv23.i.i52
  %ptridx.ascast.i.i.i54 = addrspacecast i8 addrspace(1)* %ptridx.i.i.i53 to i8 addrspace(4)*
  store i8 %24, i8 addrspace(4)* %ptridx.ascast.i.i.i54, align 1, !tbaa !9
  %inc26.i.i55 = add nuw i64 %inc26.i.i5572, 1
  %inc27.i.i56 = add i32 %inc27.i.i5673, 1
  %cmp18.i.i57 = icmp ult i64 %inc26.i.i55, %idxprom.i5
  br i1 %cmp18.i.i57, label %for.body20.i.for.body20.i_crit_edge.i60, label %for.cond.cleanup19.i.loopexit.i37, !llvm.loop !29

_ZN2cl4sycllsERKNS0_6streamEPKc.exit61:           ; preds = %for.end.i28, %lor.lhs.false.i.i33, %for.cond.cleanup19.i.i47
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %11) #8
  ret void
}

attributes #0 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="repro-1.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { convergent inlinehint norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #6 = { convergent norecurse mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { convergent nounwind readnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }
attributes #9 = { convergent }
attributes #10 = { convergent nounwind readnone }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!3}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 12.0.0 (https://AlexeySachkov@github.com/otcshare/llvm 9d0e3525ba04b7b45e7eba31feb280caf7774898)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{!6, !6, i64 0}
!6 = !{!"long", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!7, !7, i64 0}
!10 = !{!11, !6, i64 120}
!11 = !{!"_ZTSN2cl4sycl6streamE", !7, i64 0, !12, i64 16, !16, i64 48, !12, i64 80, !17, i64 112, !17, i64 116, !6, i64 120, !18, i64 128, !17, i64 132, !17, i64 136, !17, i64 140}
!12 = !{!"_ZTSN2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE", !13, i64 0, !7, i64 24}
!13 = !{!"_ZTSN2cl4sycl6detail18AccessorImplDeviceILi1EEE", !14, i64 0, !15, i64 8, !15, i64 16}
!14 = !{!"_ZTSN2cl4sycl2idILi1EEE"}
!15 = !{!"_ZTSN2cl4sycl5rangeILi1EEE"}
!16 = !{!"_ZTSN2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE", !13, i64 0, !7, i64 24}
!17 = !{!"int", !7, i64 0}
!18 = !{!"_ZTSN2cl4sycl18stream_manipulatorE", !7, i64 0}
!19 = !{!11, !17, i64 112}
!20 = !{!21, !23, !25}
!21 = distinct !{!21, !22, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!22 = distinct !{!22, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!23 = distinct !{!23, !24, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!24 = distinct !{!24, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!25 = distinct !{!25, !26, !"_ZN2cl4sycl6detail7Builder10getElementILi1EEEKNS0_2idIXT_EEEPS5_: %agg.result"}
!26 = distinct !{!26, !"_ZN2cl4sycl6detail7Builder10getElementILi1EEEKNS0_2idIXT_EEEPS5_"}
!27 = distinct !{!27, !28}
!28 = !{!"llvm.loop.mustprogress"}
!29 = distinct !{!29, !28}
!30 = !{!31}
!31 = distinct !{!31, !32, !"_ZNK2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEixILi1EEENSt9enable_ifIXaaeqT_Li1EeqcvS3_Li1029ELS3_1029EENS0_6atomicIjLNS2_13address_spaceE1EEEE4typeEm: %agg.result"}
!32 = distinct !{!32, !"_ZNK2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEixILi1EEENSt9enable_ifIXaaeqT_Li1EeqcvS3_Li1029ELS3_1029EENS0_6atomicIjLNS2_13address_spaceE1EEEE4typeEm"}
!33 = !{!34}
!34 = distinct !{!34, !35, !"_ZNK2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9get_rangeILi1EvEENS0_5rangeILi1EEEv: %agg.result"}
!35 = distinct !{!35, !"_ZNK2cl4sycl8accessorIcLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9get_rangeILi1EvEENS0_5rangeILi1EEEv"}
!36 = !{!37}
!37 = distinct !{!37, !38, !"_ZNK2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEixILi1EEENSt9enable_ifIXaaeqT_Li1EeqcvS3_Li1029ELS3_1029EENS0_6atomicIjLNS2_13address_spaceE1EEEE4typeEm: %agg.result"}
!38 = distinct !{!38, !"_ZNK2cl4sycl8accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEixILi1EEENSt9enable_ifIXaaeqT_Li1EeqcvS3_Li1029ELS3_1029EENS0_6atomicIjLNS2_13address_spaceE1EEEE4typeEm"}
!39 = distinct !{!39, !28}
!40 = !{!11, !17, i64 140}
!41 = !{!11, !17, i64 136}
!42 = !{!11, !17, i64 132}
!43 = !{!44, !44, i64 0}
!44 = !{!"float", !7, i64 0}
!45 = distinct !{!45, !28}
!46 = distinct !{!46, !28}
!47 = distinct !{!47, !28}
!48 = distinct !{!48, !28}
!49 = distinct !{!49, !28}
!50 = !{!17, !17, i64 0}
!51 = distinct !{!51, !28}
!52 = distinct !{!52, !28}
!53 = distinct !{!53, !28}
!54 = distinct !{!54, !28}
!55 = distinct !{!55, !28}
!56 = distinct !{!56, !28}
!57 = distinct !{!57, !28}
!58 = distinct !{!58, !28}
!59 = !{!60, !62, !64}
!60 = distinct !{!60, !61, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!61 = distinct !{!61, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!62 = distinct !{!62, !63, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!63 = distinct !{!63, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!64 = distinct !{!64, !65, !"_ZN2cl4sycl6detail7Builder10getElementILi1EEEKNS0_2idIXT_EEEPS5_: %agg.result"}
!65 = distinct !{!65, !"_ZN2cl4sycl6detail7Builder10getElementILi1EEEKNS0_2idIXT_EEEPS5_"}
