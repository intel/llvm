; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN: -passes=sycl-internalization --sycl-info-path %S/../kernel-fusion/kernel-info.yaml -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.array_wrapper = type { %"struct.std::array" }
%"struct.std::array" = type { [2 x %"struct.std::array.0"] }
%"struct.std::array.0" = type { [2 x %"class.sycl::_V1::vec"] }
%"class.sycl::_V1::vec" = type { <2 x i32> }

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_start_wrapper() #0 {
entry:
  %GroupID = alloca [3 x i64], align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: noinline nounwind
define spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) %group_id, i64 %wi_id, i32 %wg_size) #2 {
entry:
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_finish_wrapper() #0 {
entry:
  %GroupID = alloca [3 x i64], align 8
  ret void
}

; Function Attrs: noinline nounwind
define spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) %group_id, i64 %wi_id) #2 {
entry:
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32) #4

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32) #4

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #4

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32) #4

define spir_kernel void @fused_0(ptr addrspace(1) align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn1, ptr byval(%"class.sycl::_V1::id") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn13, ptr addrspace(1) align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn2, ptr byval(%"class.sycl::_V1::id") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn26, ptr addrspace(1) align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, ptr byval(%"class.sycl::_V1::id") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp9, ptr addrspace(1) align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn3, ptr byval(%"class.sycl::_V1::id") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn36, ptr addrspace(1) align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut, ptr byval(%"class.sycl::_V1::id") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut9) !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_type_qual !9 !kernel_arg_base_type !8 !kernel_arg_name !10 !sycl.kernel.promote !11 !sycl.kernel.promote.localsize !12 !sycl.kernel.constants !13 {
; Scenario: Test the successful local internalization of the pointer argument
; `...KernelOne__arg_accTmp`. This means the pointer argument has been replaced
; by a pointer to the local address space (address space 3), and offset-wrapping
; instructions have been introduced. This test is based on an IR dump of
; `sycl/test-e2e/KernelFusion/internalize_array_wrapper_local.cpp`.

; CHECK-LABEL: define spir_kernel void @fused_0
; CHECK-SAME: (ptr addrspace(1) align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN1:%.*]], ptr byval(%"class.sycl::_V1::id") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN13:%.*]], ptr addrspace(1) align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN2:%.*]], ptr byval(%"class.sycl::_V1::id") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN26:%.*]], ptr addrspace(3) align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP:%.*]], ptr byval(%"class.sycl::_V1::id") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP9:%.*]], ptr addrspace(1) align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCIN3:%.*]], ptr byval(%"class.sycl::_V1::id") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCIN36:%.*]], ptr addrspace(1) align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT:%.*]], ptr byval(%"class.sycl::_V1::id") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT9:%.*]]) !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_type_qual !9 !kernel_arg_base_type !8 !kernel_arg_name !10 !sycl.kernel.constants !11 {
; CHECK-NEXT:  entry:
; CHECK:         [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP93_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP9]], align 1
; CHECK:         [[TMP0:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP93_SROA_0_0_COPYLOAD]], 4
; CHECK:         [[ADD_PTR_I43_I:%.*]] = getelementptr inbounds %struct.array_wrapper, ptr addrspace(3) [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP]], i64 [[TMP0]]
; CHECK:         [[TMP1:%.*]] = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #[[ATTR4:[0-9]+]]
; CHECK:         [[TMP2:%.*]] = add i64 [[TMP1]], [[TMP0]]
; CHECK:         [[TMP3:%.*]] = urem i64 [[TMP2]], 4
; CHECK:         [[ARRAYIDX_I34_I_I:%.*]] = getelementptr inbounds %struct.array_wrapper, ptr addrspace(3) [[ADD_PTR_I43_I]], i64 [[TMP3]]
; CHECK:         [[I_0_I_I:%.*]] = phi i64 {{.*}}
; CHECK:         [[TMP4:%.*]] = add i64 0, [[TMP3]]
; CHECK:         [[TMP5:%.*]] = add i64 [[TMP4]], [[TMP0]]
; CHECK:         [[TMP6:%.*]] = urem i64 [[TMP5]], 4
; CHECK:         [[ARRAYIDX_I_I_I37_I_I:%.*]] = getelementptr inbounds [2 x %"struct.std::array.0"], ptr addrspace(3) [[ARRAYIDX_I34_I_I]], i64 [[TMP6]], i64 [[I_0_I_I]]
; CHECK:         store <2 x i32> {{%.*}}, ptr addrspace(3) [[ARRAYIDX_I_I_I37_I_I]], align 8
; CHECK:         [[ARRAYIDX_I_I39_I_I_1:%.*]] = getelementptr inbounds [2 x %"class.sycl::_V1::vec"], ptr addrspace(3) [[ARRAYIDX_I_I_I37_I_I]], i64 0, i64 1
; CHECK:         store <2 x i32> {{%.*}}, ptr addrspace(3) [[ARRAYIDX_I_I39_I_I_1]], align 8
; CHECK:         [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP94_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP9]], align 1
; CHECK:         [[TMP11:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP94_SROA_0_0_COPYLOAD]], 4
; CHECK:         [[ADD_PTR_I_I7:%.*]] = getelementptr inbounds %struct.array_wrapper, ptr addrspace(3) [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP]], i64 [[TMP11]]
; CHECK:         [[TMP12:%.*]] = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #[[ATTR4]]
; CHECK:         [[TMP13:%.*]] = add i64 [[TMP12]], [[TMP11]]
; CHECK:         [[TMP14:%.*]] = urem i64 [[TMP13]], 4
; CHECK:         [[ARRAYIDX_I_I_I11:%.*]] = getelementptr inbounds %struct.array_wrapper, ptr addrspace(3) [[ADD_PTR_I_I7]], i64 [[TMP14]]
; CHECK:         [[I_0_I_I15:%.*]] = phi i64 {{.*}}
; CHECK:         [[TMP15:%.*]] = add i64 0, [[TMP14]]
; CHECK:         [[TMP16:%.*]] = add i64 [[TMP15]], [[TMP11]]
; CHECK:         [[TMP17:%.*]] = urem i64 [[TMP16]], 4
; CHECK:         [[ARRAYIDX_I_I_I_I_I18:%.*]] = getelementptr inbounds [2 x %"struct.std::array.0"], ptr addrspace(3) [[ARRAYIDX_I_I_I11]], i64 [[TMP17]], i64 [[I_0_I_I15]]
; CHECK:         [[TMP18:%.*]] = load <2 x i32>, ptr addrspace(3) [[ARRAYIDX_I_I_I_I_I18]], align 8
; CHECK:         [[ARRAYIDX_I_I_I_I27_1:%.*]] = getelementptr inbounds [2 x %"class.sycl::_V1::vec"], ptr addrspace(3) [[ARRAYIDX_I_I_I_I_I18]], i64 0, i64 1
; CHECK:         ret void
;
entry:
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp93.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp9, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn262.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn26, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn131.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn13, align 1
  %add.ptr.i.i = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn1, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn131.sroa.0.0.copyload
  %add.ptr.i34.i = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn2, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn262.sroa.0.0.copyload
  %add.ptr.i43.i = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp93.sroa.0.0.copyload
  %0 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #4
  %cmp.i.i.i = icmp ult i64 %0, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i)
  %arrayidx.i.i.i = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %add.ptr.i.i, i64 %0
  %arrayidx.i30.i.i = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %add.ptr.i34.i, i64 %0
  %arrayidx.i34.i.i = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %add.ptr.i43.i, i64 %0
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.body.i.i, %entry
  %i.0.i.i = phi i64 [ 0, %entry ], [ %inc17.i.i, %for.body.i.i ]
  %cmp.i.i = icmp ult i64 %i.0.i.i, 16
  br i1 %cmp.i.i, label %for.body.i.i, label %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne.exit

for.body.i.i:                                     ; preds = %for.cond.i.i
  %arrayidx.i.i.i.i.i = getelementptr inbounds [2 x %"struct.std::array.0"], ptr addrspace(1) %arrayidx.i.i.i, i64 0, i64 %i.0.i.i
  %arrayidx.i.i.i36.i.i = getelementptr inbounds [2 x %"struct.std::array.0"], ptr addrspace(1) %arrayidx.i30.i.i, i64 0, i64 %i.0.i.i
  %arrayidx.i.i.i37.i.i = getelementptr inbounds [2 x %"struct.std::array.0"], ptr addrspace(1) %arrayidx.i34.i.i, i64 0, i64 %i.0.i.i
  %1 = load <2 x i32>, ptr addrspace(1) %arrayidx.i.i.i.i.i, align 8
  %2 = load <2 x i32>, ptr addrspace(1) %arrayidx.i.i.i36.i.i, align 8
  %add.i.i.i = add <2 x i32> %1, %2
  store <2 x i32> %add.i.i.i, ptr addrspace(1) %arrayidx.i.i.i37.i.i, align 8
  %arrayidx.i.i.i.i.1 = getelementptr inbounds [2 x %"class.sycl::_V1::vec"], ptr addrspace(1) %arrayidx.i.i.i.i.i, i64 0, i64 1
  %arrayidx.i.i38.i.i.1 = getelementptr inbounds [2 x %"class.sycl::_V1::vec"], ptr addrspace(1) %arrayidx.i.i.i36.i.i, i64 0, i64 1
  %3 = load <2 x i32>, ptr addrspace(1) %arrayidx.i.i.i.i.1, align 8
  %4 = load <2 x i32>, ptr addrspace(1) %arrayidx.i.i38.i.i.1, align 8
  %add.i.i.i.1 = add <2 x i32> %3, %4
  %arrayidx.i.i39.i.i.1 = getelementptr inbounds [2 x %"class.sycl::_V1::vec"], ptr addrspace(1) %arrayidx.i.i.i37.i.i, i64 0, i64 1
  store <2 x i32> %add.i.i.i.1, ptr addrspace(1) %arrayidx.i.i39.i.i.1, align 8
  %inc17.i.i = add nuw nsw i64 %i.0.i.i, 1
  br label %for.cond.i.i

_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne.exit: ; preds = %for.cond.i.i
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut96.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut9, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn365.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn36, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp94.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp9, align 1
  %add.ptr.i.i7 = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp94.sroa.0.0.copyload
  %add.ptr.i34.i8 = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn3, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn365.sroa.0.0.copyload
  %add.ptr.i43.i9 = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut96.sroa.0.0.copyload
  %5 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #4
  %cmp.i.i.i10 = icmp ult i64 %5, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i10)
  %arrayidx.i.i.i11 = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %add.ptr.i.i7, i64 %5
  %arrayidx.i30.i.i12 = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %add.ptr.i34.i8, i64 %5
  %arrayidx.i34.i.i13 = getelementptr inbounds %struct.array_wrapper, ptr addrspace(1) %add.ptr.i43.i9, i64 %5
  br label %for.cond.i.i14

for.cond.i.i14:                                   ; preds = %for.body.i.i17, %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne.exit
  %i.0.i.i15 = phi i64 [ 0, %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne.exit ], [ %inc17.i.i25, %for.body.i.i17 ]
  %cmp.i.i16 = icmp ult i64 %i.0.i.i15, 16
  br i1 %cmp.i.i16, label %for.body.i.i17, label %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo.exit

for.body.i.i17:                                   ; preds = %for.cond.i.i14
  %arrayidx.i.i.i.i.i18 = getelementptr inbounds [2 x %"struct.std::array.0"], ptr addrspace(1) %arrayidx.i.i.i11, i64 0, i64 %i.0.i.i15
  %arrayidx.i.i.i36.i.i19 = getelementptr inbounds [2 x %"struct.std::array.0"], ptr addrspace(1) %arrayidx.i30.i.i12, i64 0, i64 %i.0.i.i15
  %arrayidx.i.i.i37.i.i20 = getelementptr inbounds [2 x %"struct.std::array.0"], ptr addrspace(1) %arrayidx.i34.i.i13, i64 0, i64 %i.0.i.i15
  %6 = load <2 x i32>, ptr addrspace(1) %arrayidx.i.i.i.i.i18, align 8
  %7 = load <2 x i32>, ptr addrspace(1) %arrayidx.i.i.i36.i.i19, align 8
  %mul.i.i.i = mul <2 x i32> %6, %7
  store <2 x i32> %mul.i.i.i, ptr addrspace(1) %arrayidx.i.i.i37.i.i20, align 8
  %arrayidx.i.i.i.i27.1 = getelementptr inbounds [2 x %"class.sycl::_V1::vec"], ptr addrspace(1) %arrayidx.i.i.i.i.i18, i64 0, i64 1
  %arrayidx.i.i38.i.i28.1 = getelementptr inbounds [2 x %"class.sycl::_V1::vec"], ptr addrspace(1) %arrayidx.i.i.i36.i.i19, i64 0, i64 1
  %8 = load <2 x i32>, ptr addrspace(1) %arrayidx.i.i.i.i27.1, align 8
  %9 = load <2 x i32>, ptr addrspace(1) %arrayidx.i.i38.i.i28.1, align 8
  %mul.i.i.i.1 = mul <2 x i32> %8, %9
  %arrayidx.i.i39.i.i29.1 = getelementptr inbounds [2 x %"class.sycl::_V1::vec"], ptr addrspace(1) %arrayidx.i.i.i37.i.i20, i64 0, i64 1
  store <2 x i32> %mul.i.i.i.1, ptr addrspace(1) %arrayidx.i.i39.i.i29.1, align 8
  %inc17.i.i25 = add nuw nsw i64 %i.0.i.i15, 1
  br label %for.cond.i.i14

_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo.exit: ; preds = %for.cond.i.i14
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #5

attributes #0 = { alwaysinline nounwind }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { noinline nounwind }
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
!6 = !{i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0}
!7 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!8 = !{!"struct array_wrapper*", !"class.sycl::_V1::id", !"struct array_wrapper*", !"class.sycl::_V1::id", !"struct array_wrapper*", !"class.sycl::_V1::id", !"struct array_wrapper*", !"class.sycl::_V1::id", !"struct array_wrapper*", !"class.sycl::_V1::id"}
!9 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!10 = !{!"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn1", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn13", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn2", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn26", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp9", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn3", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn36", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut9"}
!11 = !{!"none", !"none", !"none", !"none", !"local", !"none", !"none", !"none", !"none", !"none"}
!12 = !{!"", !"", !"", !"", i64 4, !"", !"", !"", !"", !""}
!13 = !{!"", !"\00\00\00\00\00\00\00\00", !"", !"\00\00\00\00\00\00\00\00", !"", !"", !"", !"\00\00\00\00\00\00\00\00", !"", !"\00\00\00\00\00\00\00\00"}
