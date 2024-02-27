; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN:   -passes=sycl-internalization -S %s | FileCheck %s

; This test is a reduced IR version of
; sycl/test-e2e/KernelFusion/internalize_non_unit_localsize.cpp

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.MyStruct = type { i32, %"class.sycl::_V1::vec" }
%"class.sycl::_V1::vec" = type { <3 x i32> }

declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32) local_unnamed_addr #1

define spir_kernel void @fused_0(ptr addrspace(1) nocapture align 16 %KernelOne__arg_accTmp,
   ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 %KernelOne__arg_accTmp3,
   ptr addrspace(1) nocapture readonly align 4 %KernelOne__arg_accIn,
   ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 %KernelOne__arg_accIn6,
   ptr addrspace(1) nocapture writeonly align 1 %KernelOne__arg_accTmp27,
   ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 %KernelOne__arg_accTmp210,
   ptr addrspace(1) nocapture writeonly align 4 %KernelTwo__arg_accOut,
   ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 %KernelTwo__arg_accOut3)
     local_unnamed_addr #2 !sycl.kernel.promote !11 !sycl.kernel.promote.localsize !12 !sycl.kernel.promote.elemsize !13 {
; CHECK-LABEL: define spir_kernel void @fused_0(
; CHECK-SAME: ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 [[KERNELONE__ARG_ACCTMP3:%[^,]*accTmp3]],
; CHECK-SAME: ptr nocapture readonly byval(%"class.sycl::_V1::range") align 8 [[KERNELONE__ARG_ACCTMP210:%[^,]*accTmp210]]
; CHECK:         entry:
; CHECK:           [[TMP0:%.*]] = alloca i8, i64 3, align 1
; CHECK:           [[TMP1:%.*]] = alloca i8, i64 96, align 16
; CHECK:           [[KERNELONE__ARG_ACCTMP2103_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[KERNELONE__ARG_ACCTMP210]], align 8
; CHECK:           [[KERNELONE__ARG_ACCTMP31_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[KERNELONE__ARG_ACCTMP3]], align 8
; CHECK:           [[TMP2:%.*]] = urem i64 [[KERNELONE__ARG_ACCTMP31_SROA_0_0_COPYLOAD]], 3
; CHECK:           [[TMP3:%.*]] = urem i64 [[KERNELONE__ARG_ACCTMP2103_SROA_0_0_COPYLOAD]], 3
; CHECK:           [[TMP4:%.*]] = tail call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0)
; CHECK:           [[MUL:%.*]] = mul nuw nsw i64 [[TMP4]], 3
; CHECK:           [[ADD:%.*]] = add nuw nsw i64 [[MUL]], 1
; CHECK:           [[TMP6:%.*]] = add i64 [[TMP2]], [[MUL]]
; CHECK:           [[TMP7:%.*]] = urem i64 [[TMP6]], 3
; CHECK:           [[ARRAYIDX_1:%.*]] = getelementptr inbounds %struct.MyStruct, ptr [[TMP1]], i64 [[TMP7]]
; CHECK:           [[ADDA:%.*]] = add i64 [[TMP7]], 1
; CHECK:           [[TMP7A:%.*]] = urem i64 [[ADDA]], 3

; COM:             This constant i8-GEP was rewritten to encode an _element_ offset, and subsequently remapped.
; CHECK:           [[ARRAYIDX_1A:%.*]] = getelementptr inbounds i256, ptr [[TMP1]], i64 [[TMP7A]]

; COM:             This i8-GEP _was_ not remapped because it addresses into a single MyStruct element
; CHECK:           [[ARRAYIDX_2:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_1A]], i64 20

; CHECK:           store i32 {{.*}}, ptr [[ARRAYIDX_2]], align 4
; CHECK:           [[TMP8:%.*]] = add i64 [[TMP3]], [[ADD]]
; CHECK:           [[TMP9:%.*]] = urem i64 [[TMP8]], 3

; COM:             This i8-GEP was remapped because it selects an element of the underlying i8-buffer
; CHECK:           [[ARRAYIDX_3:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP9]]

; CHECK:           store i8 {{.*}}, ptr [[ARRAYIDX_3]], align 1
; CHECK:           store i32 {{.*}}, ptr addrspace(1)
; CHECK:           ret void
;
entry:
  %KernelOne__arg_accTmp2103.sroa.0.0.copyload = load i64, ptr %KernelOne__arg_accTmp210, align 8
  %KernelOne__arg_accIn62.sroa.0.0.copyload = load i64, ptr %KernelOne__arg_accIn6, align 8
  %KernelOne__arg_accTmp31.sroa.0.0.copyload = load i64, ptr %KernelOne__arg_accTmp3, align 8
  %add.ptr.j2 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %KernelOne__arg_accTmp, i64 %KernelOne__arg_accTmp31.sroa.0.0.copyload
  %add.ptr.i35.i = getelementptr inbounds i32, ptr addrspace(1) %KernelOne__arg_accIn, i64 %KernelOne__arg_accIn62.sroa.0.0.copyload
  %add.ptr.i44.i = getelementptr inbounds i8, ptr addrspace(1) %KernelOne__arg_accTmp27, i64 %KernelOne__arg_accTmp2103.sroa.0.0.copyload
  %0 = tail call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #3
  %mul.j2 = mul nuw nsw i64 %0, 3
  %add.j2 = add nuw nsw i64 %mul.j2, 1
  %arrayidx.j2 = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i35.i, i64 %add.j2
  %1 = load i32, ptr addrspace(1) %arrayidx.j2, align 4
  %arrayidx.i54.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.j2, i64 %mul.j2
  ; Mimic %add.j2 by artificially representing it as a constant byte offset (sizeof(MyStruct)==32 byte)
  %arrayidx.plus.one.element = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i54.i, i64 32
  %arrayidx.j3 = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.plus.one.element, i64 20
  store i32 %1, ptr addrspace(1) %arrayidx.j3, align 4
  %conv.j2 = trunc i32 %1 to i8
  %arrayidx.i70.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i44.i, i64 %add.j2
  store i8 %conv.j2, ptr addrspace(1) %arrayidx.i70.i, align 1
  %KernelTwo__arg_accOut34.sroa.0.0.copyload = load i64, ptr %KernelTwo__arg_accOut3, align 8
  %add.ptr.i.i7 = getelementptr inbounds i32, ptr addrspace(1) %KernelTwo__arg_accOut, i64 %KernelTwo__arg_accOut34.sroa.0.0.copyload
  %2 = load i32, ptr addrspace(1) %arrayidx.j3, align 4
  %conv.i.i13 = sext i8 %conv.j2 to i32
  %add.i.i14 = add nsw i32 %2, %conv.i.i13
  %arrayidx.i62.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i7, i64 %add.j2
  store i32 %add.i.i14, ptr addrspace(1) %arrayidx.i62.i, align 4
  ret void
}

attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: write) }
attributes #3 = { nounwind willreturn memory(none) }

!11 = !{!"private", !"none", !"none", !"none", !"private", !"none", !"none", !"none"}
!12 = !{i64 3, !"", !"", !"", i64 3, !"", !"", !""}
!13 = !{i64 32, !"", !"", !"", i64 1, !"", !"", !""}
