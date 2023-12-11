; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN:   -passes=sycl-internalization --sycl-info-path %S/../kernel-fusion/kernel-info.yaml -S %s | FileCheck %s

; This test is the IR version of sycl/test-e2e/KernelFusion/internalize_non_unit_localsize.cpp

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.MyStruct = type { i32, %"class.sycl::_V1::vec" }
%"class.sycl::_V1::vec" = type { <3 x i32> }

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_start_wrapper() #0 {
entry:
  %GroupID = alloca [3 x i64], align 8, !spirv.Decorations !6
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg %0, ptr nocapture %1) #1

; Function Attrs: noinline nounwind optnone
define spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) %group_id, i64 %wi_id, i32 %wg_size) #2 {
entry:
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg %0, ptr nocapture %1) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef %0) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_finish_wrapper() #0 {
entry:
  %GroupID = alloca [3 x i64], align 8, !spirv.Decorations !6
  ret void
}

; Function Attrs: noinline nounwind optnone
define spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) %group_id, i64 %wi_id) #2 {
entry:
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 %0) #4

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 %0) #4

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #4

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 %0) #4

define spir_kernel void @fused_0(ptr addrspace(1) align 16 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, ptr byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3, ptr addrspace(1) align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn, ptr byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6, ptr addrspace(1) align 1 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27, ptr byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210, ptr addrspace(1) align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut, ptr byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3, ptr byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn9) !kernel_arg_addr_space !8 !kernel_arg_access_qual !9 !kernel_arg_type !10 !kernel_arg_type_qual !11 !kernel_arg_base_type !10 !kernel_arg_name !12 !sycl.kernel.promote !13 !sycl.kernel.promote.localsize !14 !sycl.kernel.promote.elemsize !15 !sycl.kernel.constants !16 {
; CHECK-LABEL: define spir_kernel void @fused_0(
; CHECK-SAME:      ptr byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP3:%.*]], ptr addrspace(1) align 4 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN:%.*]], ptr byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN6:%.*]], ptr byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP210:%.*]], ptr addrspace(1) align 4 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT:%.*]], ptr byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT3:%.*]], ptr byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCIN9:%.*]]) !kernel_arg_addr_space !8 !kernel_arg_access_qual !9 !kernel_arg_type !10 !kernel_arg_type_qual !11 !kernel_arg_base_type !10 !kernel_arg_name !12 !sycl.kernel.constants [[META13:![0-9]+]] {
; CHECK:         entry:
; CHECK:           [[TMP0:%.*]] = alloca i8, i64 3, align 1
; CHECK:           [[TMP1:%.*]] = alloca i8, i64 96, align 16
; CHECK:           [[TMP2:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ADD_PTR_I_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT:%.*]], ptr [[TMP1]], i64 [[TMP2]]
; CHECK:           [[TMP3:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ADD_PTR_I44_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP3]]
; CHECK:           [[TMP7:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I51_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP7]]
; CHECK:           [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I51_I]], i64 20
; CHECK:           [[TMP10:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I66_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP10]]
; CHECK:           [[ARRAYIDX_I_I68_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I66_I]], i64 20
; CHECK:           [[TMP13:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I83_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP13]]
; CHECK:           [[ARRAYIDX_I_I85_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I83_I]], i64 20
; CHECK:           [[TMP17:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I98_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP17]]
; CHECK:           [[TMP21:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I114_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP21]]
; CHECK:           [[TMP25:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I128_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP25]]
; CHECK:           [[TMP26:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ADD_PTR_I41_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP26]]
; CHECK:           [[TMP27:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ADD_PTR_I59_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP27]]
; CHECK:           [[TMP30:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I_I10:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP30]]
; CHECK:           [[ARRAYIDX_I_I_I11:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I_I10]], i64 20
; CHECK:           [[TMP34:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I75_I14:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP34]]
; CHECK:           [[TMP37:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I89_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP37]]
; CHECK:           [[ARRAYIDX_I_I91_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I89_I]], i64 20
; CHECK:           [[TMP41:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I106_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP41]]
; CHECK:           [[TMP44:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I122_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP44]]
; CHECK:           [[ARRAYIDX_I_I124_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I122_I]], i64 20
; CHECK:           [[TMP48:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I137_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP48]]
; CHECK:           ret void
;
entry:
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2103.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn62.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3, align 1
  %add.ptr.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31.sroa.0.0.copyload
  %add.ptr.i35.i = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn62.sroa.0.0.copyload
  %add.ptr.i44.i = getelementptr inbounds i8, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2103.sroa.0.0.copyload
  %0 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #4
  %cmp.i.i.i = icmp ult i64 %0, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i)
  %mul.i.i = mul nuw nsw i64 %0, 3, !spirv.Decorations !17
  %arrayidx.i.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i35.i, i64 %mul.i.i
  %1 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4
  %arrayidx.i51.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %mul.i.i
  %arrayidx.i.i.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i51.i, i64 20
  store i32 %1, ptr addrspace(1) %arrayidx.i.i.i, align 4
  %add.i.i = add nuw nsw i64 %mul.i.i, 1, !spirv.Decorations !17
  %arrayidx.i58.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i35.i, i64 %add.i.i
  %2 = load i32, ptr addrspace(1) %arrayidx.i58.i, align 4
  %arrayidx.i66.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i.i
  %arrayidx.i.i68.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i66.i, i64 20
  store i32 %2, ptr addrspace(1) %arrayidx.i.i68.i, align 4
  %add.i72.i = add nuw nsw i64 %mul.i.i, 2, !spirv.Decorations !17
  %arrayidx.i75.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i35.i, i64 %add.i72.i
  %3 = load i32, ptr addrspace(1) %arrayidx.i75.i, align 4
  %arrayidx.i83.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i72.i
  %arrayidx.i.i85.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i83.i, i64 20
  store i32 %3, ptr addrspace(1) %arrayidx.i.i85.i, align 4
  %4 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4
  %5 = trunc i32 %4 to i8
  %conv.i.i = xor i8 %5, -86
  %arrayidx.i98.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i44.i, i64 %add.i72.i
  store i8 %conv.i.i, ptr addrspace(1) %arrayidx.i98.i, align 1
  %6 = load i32, ptr addrspace(1) %arrayidx.i58.i, align 4
  %7 = trunc i32 %6 to i8
  %conv50.i.i = xor i8 %7, -86
  %arrayidx.i114.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i44.i, i64 %add.i.i
  store i8 %conv50.i.i, ptr addrspace(1) %arrayidx.i114.i, align 1
  %8 = load i32, ptr addrspace(1) %arrayidx.i75.i, align 4
  %9 = trunc i32 %8 to i8
  %conv64.i.i = xor i8 %9, -86
  %arrayidx.i128.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i44.i, i64 %mul.i.i
  store i8 %conv64.i.i, ptr addrspace(1) %arrayidx.i128.i, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2107.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn96.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn9, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp35.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut34.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3, align 1
  %add.ptr.i.i8 = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut34.sroa.0.0.copyload
  %add.ptr.i41.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp35.sroa.0.0.copyload
  %add.ptr.i50.i = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn96.sroa.0.0.copyload
  %add.ptr.i59.i = getelementptr inbounds i8, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2107.sroa.0.0.copyload
  %10 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #4
  %cmp.i.i.i9 = icmp ult i64 %10, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i9)
  %mul.i61.i = mul nuw nsw i64 %10, 3, !spirv.Decorations !17
  %arrayidx.i.i10 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i41.i, i64 %mul.i61.i
  %arrayidx.i.i.i11 = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i.i10, i64 20
  %11 = load i32, ptr addrspace(1) %arrayidx.i.i.i11, align 4
  %arrayidx.i67.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i50.i, i64 %mul.i61.i
  %12 = load i32, ptr addrspace(1) %arrayidx.i67.i, align 4
  %mul.i.i12 = mul nsw i32 %11, %12, !spirv.Decorations !20
  %add.i72.i13 = add nuw nsw i64 %mul.i61.i, 2, !spirv.Decorations !17
  %arrayidx.i75.i14 = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i59.i, i64 %add.i72.i13
  %13 = load i8, ptr addrspace(1) %arrayidx.i75.i14, align 1
  %conv.i.i15 = sext i8 %13 to i32
  %add.i.i16 = add nsw i32 %mul.i.i12, %conv.i.i15, !spirv.Decorations !20
  %arrayidx.i81.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i8, i64 %mul.i61.i
  store i32 %add.i.i16, ptr addrspace(1) %arrayidx.i81.i, align 4
  %add.i86.i = add nuw nsw i64 %mul.i61.i, 1, !spirv.Decorations !17
  %arrayidx.i89.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i41.i, i64 %add.i86.i
  %arrayidx.i.i91.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i89.i, i64 20
  %14 = load i32, ptr addrspace(1) %arrayidx.i.i91.i, align 4
  %arrayidx.i98.i17 = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i50.i, i64 %add.i86.i
  %15 = load i32, ptr addrspace(1) %arrayidx.i98.i17, align 4
  %mul28.i.i = mul nsw i32 %14, %15, !spirv.Decorations !20
  %arrayidx.i106.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i59.i, i64 %add.i86.i
  %16 = load i8, ptr addrspace(1) %arrayidx.i106.i, align 1
  %conv35.i.i = sext i8 %16 to i32
  %add36.i.i = add nsw i32 %mul28.i.i, %conv35.i.i, !spirv.Decorations !20
  %arrayidx.i114.i18 = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i8, i64 %add.i86.i
  store i32 %add36.i.i, ptr addrspace(1) %arrayidx.i114.i18, align 4
  %arrayidx.i122.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i41.i, i64 %add.i72.i13
  %arrayidx.i.i124.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i122.i, i64 20
  %17 = load i32, ptr addrspace(1) %arrayidx.i.i124.i, align 4
  %arrayidx.i131.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i50.i, i64 %add.i72.i13
  %18 = load i32, ptr addrspace(1) %arrayidx.i131.i, align 4
  %mul57.i.i = mul nsw i32 %17, %18, !spirv.Decorations !20
  %arrayidx.i137.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i59.i, i64 %mul.i61.i
  %19 = load i8, ptr addrspace(1) %arrayidx.i137.i, align 1
  %conv62.i.i = sext i8 %19 to i32
  %add63.i.i = add nsw i32 %mul57.i.i, %conv62.i.i, !spirv.Decorations !20
  %arrayidx.i145.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i8, i64 %add.i72.i13
  store i32 %add63.i.i, ptr addrspace(1) %arrayidx.i145.i, align 4
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly %0, ptr noalias nocapture readonly %1, i64 %2, i1 immarg %3) #5

attributes #0 = { alwaysinline nounwind }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { noinline nounwind optnone }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #4 = { nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!spirv.Generator = !{!5}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i32 1, i32 2}
!3 = !{i32 1, i32 0}
!4 = !{}
!5 = !{i16 6, i16 14}
!6 = !{!7}
!7 = !{i32 44, i32 8}
!8 = !{i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0}
!9 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!10 = !{!"struct MyStruct*", !"class.sycl::_V1::range", !"int*", !"class.sycl::_V1::range", !"char*", !"class.sycl::_V1::range", !"int*", !"class.sycl::_V1::range", !"class.sycl::_V1::range"}
!11 = !{!"", !"", !"", !"", !"", !"", !"", !"", !""}
!12 = !{!"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn9"}
!13 = !{!"private", !"none", !"none", !"none", !"private", !"none", !"none", !"none", !"none"}
!14 = !{i64 3, !"", !"", !"", i64 3, !"", !"", !"", !""}
!15 = !{i64 32, !"", !"", !"", i64 1, !"", !"", !"", !""}
!16 = !{!"", !"", !"", !"\00\00\00\00\00\00\00\00", !"", !"", !"", !"\00\00\00\00\00\00\00\00", !"\00\00\00\00\00\00\00\00"}
!17 = !{!18, !19}
!18 = !{i32 4469}
!19 = !{i32 4470}
!20 = !{!18}
