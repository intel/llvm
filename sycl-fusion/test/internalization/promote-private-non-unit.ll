; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN:   -passes=sycl-internalization --sycl-info-path %S/../kernel-fusion/kernel-info.yaml -S %s | FileCheck %s

; This test is a reduced IR version of
; sycl/test-e2e/KernelFusion/internalize_non_unit_localsize.cpp

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.MyStruct = type { i32, %"class.sycl::_V1::vec" }
%"class.sycl::_V1::vec" = type { <3 x i32> }

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: write)
define spir_kernel void @fused_0(ptr addrspace(1) nocapture align 16 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp,
   ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3,
   ptr addrspace(1) nocapture readonly align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn,
   ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6,
   ptr addrspace(1) nocapture writeonly align 1 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27,
   ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210,
   ptr addrspace(1) nocapture writeonly align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut,
   ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3)
     local_unnamed_addr #2 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_type_qual !9 !kernel_arg_base_type !8 !kernel_arg_name !10 !sycl.kernel.promote !11 !sycl.kernel.promote.localsize !12 !sycl.kernel.promote.elemsize !13 !sycl.kernel.constants !14 {
; CHECK-LABEL: define spir_kernel void @fused_0(
; CHECK-SAME: ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP3:%[^,]*accTmp3]],
; CHECK-SAME: ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP210:%[^,]*accTmp210]]
; CHECK:         entry:
; CHECK:           [[TMP0:%.*]] = alloca i8, i64 3, align 1
; CHECK:           [[TMP1:%.*]] = alloca i8, i64 96, align 16
; CHECK:           [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2103_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP210]], align 8
; CHECK:           [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP31_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP3]], align 8
; CHECK:           [[TMP2:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP31_SROA_0_0_COPYLOAD]], 3
; CHECK:           [[TMP3:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2103_SROA_0_0_COPYLOAD]], 3
; CHECK:           [[TMP4:%.*]] = tail call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0)
; CHECK:           [[MUL_I_I:%.*]] = mul nuw nsw i64 [[TMP4]], 3
; CHECK:           [[ADD_I_I:%.*]] = add nuw nsw i64 [[MUL_I_I]], 1
; CHECK:           [[TMP6:%.*]] = add i64 [[TMP2]], [[ADD_I_I]]
; CHECK:           [[TMP7:%.*]] = urem i64 [[TMP6]], 3
; CHECK:           [[ARRAYIDX_I54_I:%.*]] = getelementptr inbounds %struct.MyStruct, ptr [[TMP1]], i64 [[TMP7]]

; COM:             This i8-GEP _was_ not remapped because it addresses into a single MyStruct element
; CHECK:           [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I54_I]], i64 20

; CHECK:           store i32 {{.*}}, ptr [[ARRAYIDX_I_I_I]], align 4
; CHECK:           [[TMP8:%.*]] = add i64 [[TMP3]], [[ADD_I_I]]
; CHECK:           [[TMP9:%.*]] = urem i64 [[TMP8]], 3

; COM:             This i8-GEP was remapped because it selects an element of the underlying i8-buffer
; CHECK:           [[ARRAYIDX_I70_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP9]]

; CHECK:           store i8 {{.*}}, ptr [[ARRAYIDX_I70_I]], align 1
; CHECK:           store i32 {{.*}}, ptr addrspace(1)
; CHECK:           ret void
;
entry:
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2103.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210, align 8
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn62.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6, align 8
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3, align 8
  %add.ptr.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31.sroa.0.0.copyload
  %add.ptr.i35.i = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn62.sroa.0.0.copyload
  %add.ptr.i44.i = getelementptr inbounds i8, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2103.sroa.0.0.copyload
  %0 = tail call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #3
  %cmp.i.i.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i.i)
  %mul.i.i = mul nuw nsw i64 %0, 3, !spirv.Decorations !15
  %add.i.i = add nuw nsw i64 %mul.i.i, 1, !spirv.Decorations !15
  %arrayidx.i.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i35.i, i64 %add.i.i
  %1 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4
  %arrayidx.i54.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i.i
  %arrayidx.i.i.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i54.i, i64 20
  store i32 %1, ptr addrspace(1) %arrayidx.i.i.i, align 4
  %conv.i.i = trunc i32 %1 to i8
  %arrayidx.i70.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i44.i, i64 %add.i.i
  store i8 %conv.i.i, ptr addrspace(1) %arrayidx.i70.i, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut34.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3, align 8
  %add.ptr.i.i7 = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut34.sroa.0.0.copyload
  %2 = load i32, ptr addrspace(1) %arrayidx.i.i.i, align 4
  %conv.i.i13 = sext i8 %conv.i.i to i32
  %add.i.i14 = add nsw i32 %2, %conv.i.i13, !spirv.Decorations !18
  %arrayidx.i62.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i7, i64 %add.i.i
  store i32 %add.i.i14, ptr addrspace(1) %arrayidx.i62.i, align 4
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: write) }
attributes #3 = { nounwind willreturn memory(none) }

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
!6 = !{i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0}
!7 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!8 = !{!"struct MyStruct*", !"class.sycl::_V1::range", !"int*", !"class.sycl::_V1::range", !"char*", !"class.sycl::_V1::range", !"int*", !"class.sycl::_V1::range"}
!9 = !{!"", !"", !"", !"", !"", !"", !"", !""}
!10 = !{!"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3"}
!11 = !{!"private", !"none", !"none", !"none", !"private", !"none", !"none", !"none"}
!12 = !{i64 3, !"", !"", !"", i64 3, !"", !"", !""}
!13 = !{i64 32, !"", !"", !"", i64 1, !"", !"", !""}
!14 = !{!"", !"", !"", !"\00\00\00\00\00\00\00\00", !"", !"", !"", !"\00\00\00\00\00\00\00\00"}
!15 = !{!16, !17}
!16 = !{i32 4469}
!17 = !{i32 4470}
!18 = !{!16}
